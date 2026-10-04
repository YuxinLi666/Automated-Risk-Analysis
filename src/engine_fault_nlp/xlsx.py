"""Small read-only XLSX reader implemented with the Python standard library.

It intentionally handles only cell values needed by the pipeline. It never
modifies the source workbook and does not evaluate formulas.
"""

from __future__ import annotations

import posixpath
import re
import zipfile
from typing import Dict, Iterator, List, Optional, Tuple
from xml.etree import ElementTree as ET


_CELL_REF = re.compile(r"([A-Z]+)(\d+)")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def column_index(reference: str) -> int:
    match = _CELL_REF.match(reference)
    if not match:
        return 0
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - 64
    return value - 1


class XlsxReader:
    def __init__(self, path: str):
        self.path = path
        self._zip = zipfile.ZipFile(path)
        self.shared_strings = self._load_shared_strings()
        self.sheets = self._load_sheets()

    def close(self) -> None:
        self._zip.close()

    def __enter__(self) -> "XlsxReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _load_shared_strings(self) -> List[str]:
        if "xl/sharedStrings.xml" not in self._zip.namelist():
            return []
        root = ET.fromstring(self._zip.read("xl/sharedStrings.xml"))
        strings = []
        for item in root:
            if _local(item.tag) != "si":
                continue
            strings.append("".join(node.text or "" for node in item.iter() if _local(node.tag) == "t"))
        return strings

    def _load_sheets(self) -> Dict[str, str]:
        workbook = ET.fromstring(self._zip.read("xl/workbook.xml"))
        rels_root = ET.fromstring(self._zip.read("xl/_rels/workbook.xml.rels"))
        relationships = {}
        for rel in rels_root:
            relationships[rel.attrib.get("Id", "")] = rel.attrib.get("Target", "")
        sheets = {}
        for node in workbook.iter():
            if _local(node.tag) != "sheet":
                continue
            name = node.attrib.get("name", "")
            rel_id = next((v for k, v in node.attrib.items() if _local(k) == "id"), "")
            target = relationships.get(rel_id, "")
            target = target.lstrip("/")
            if not target.startswith("xl/"):
                target = posixpath.normpath(posixpath.join("xl", target))
            sheets[name] = target
        return sheets

    def sheet_names(self) -> List[str]:
        return list(self.sheets)

    def iter_rows(self, sheet_name: str) -> Iterator[Tuple[int, List[str]]]:
        target = self.sheets.get(sheet_name)
        if not target:
            raise KeyError("Worksheet not found: {}".format(sheet_name))
        with self._zip.open(target) as handle:
            for event, elem in ET.iterparse(handle, events=("end",)):
                if _local(elem.tag) != "row":
                    continue
                row_number = int(elem.attrib.get("r", "0") or 0)
                values: List[str] = []
                for cell in elem:
                    if _local(cell.tag) != "c":
                        continue
                    idx = column_index(cell.attrib.get("r", "A1"))
                    while len(values) <= idx:
                        values.append("")
                    values[idx] = self._cell_value(cell)
                yield row_number, values
                elem.clear()

    def _cell_value(self, cell: ET.Element) -> str:
        cell_type = cell.attrib.get("t", "")
        if cell_type == "inlineStr":
            return "".join(node.text or "" for node in cell.iter() if _local(node.tag) == "t")
        value: Optional[str] = None
        for child in cell:
            if _local(child.tag) == "v":
                value = child.text or ""
                break
        if value is None:
            return ""
        if cell_type == "s":
            try:
                return self.shared_strings[int(value)]
            except (IndexError, ValueError):
                return value
        if cell_type == "b":
            return "true" if value == "1" else "false"
        return value

    def rows_as_dicts(self, sheet_name: str) -> Iterator[Tuple[int, Dict[str, str]]]:
        rows = self.iter_rows(sheet_name)
        try:
            _, headers = next(rows)
        except StopIteration:
            return
        for row_number, values in rows:
            record = {}
            for idx, header in enumerate(headers):
                if not header:
                    continue
                key = header if header not in record else "{}__{}".format(header, idx + 1)
                record[key] = values[idx] if idx < len(values) else ""
            yield row_number, record

