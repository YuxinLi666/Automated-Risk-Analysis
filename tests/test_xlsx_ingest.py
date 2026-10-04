import tempfile
import unittest
import zipfile
from pathlib import Path

from engine_fault_nlp.ingest import aggregate_work_orders
from engine_fault_nlp.xlsx import XlsxReader


HEADERS = [
    "工单：序号", "故障日期", "发动机", "发动机系列", "底盘号", "故障: 行驶里程",
    "故障: 故障名称", "故障: 故障描述", "故障: 故障原因", "故障: 解决方案",
    "旧零件", "旧零件名称", "元凶件", "故障: 故障诊断码",
]


def make_xlsx(path):
    strings = HEADERS + [
        "G1", "发动机无力", "故障灯亮", "传感器损坏", "更换传感器", "P1", "压力传感器", "是", "0123",
        "P2", "温度传感器",
    ]
    shared = "<?xml version='1.0' encoding='UTF-8'?><sst xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'>" + "".join(
        "<si><t>{}</t></si>".format(value) for value in strings
    ) + "</sst>"

    def cell(column, row, value_index):
        return "<c r='{}{}' t='s'><v>{}</v></c>".format(column, row, value_index)

    columns = [chr(65 + i) for i in range(len(HEADERS))]
    header = "".join(cell(column, 1, i) for i, column in enumerate(columns))
    row1_values = [14, "", "", "", "", "", 15, 16, 17, 18, 19, 20, 21, 22]
    row2_values = [14, "", "", "", "", "", 15, 16, 17, 18, 23, 24, 21, 22]
    row1 = "".join(cell(columns[i], 2, value) for i, value in enumerate(row1_values) if value != "")
    row2 = "".join(cell(columns[i], 3, value) for i, value in enumerate(row2_values) if value != "")
    sheet = "<?xml version='1.0' encoding='UTF-8'?><worksheet xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'><sheetData><row r='1'>{}</row><row r='2'>{}</row><row r='3'>{}</row></sheetData></worksheet>".format(header, row1, row2)
    workbook = "<?xml version='1.0' encoding='UTF-8'?><workbook xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main' xmlns:r='http://schemas.openxmlformats.org/officeDocument/2006/relationships'><sheets><sheet name='Report (2)' sheetId='1' r:id='rId1'/></sheets></workbook>"
    rels = "<?xml version='1.0' encoding='UTF-8'?><Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'><Relationship Id='rId1' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet' Target='worksheets/sheet1.xml'/></Relationships>"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", rels)
        archive.writestr("xl/sharedStrings.xml", shared)
        archive.writestr("xl/worksheets/sheet1.xml", sheet)


class XlsxIngestTest(unittest.TestCase):
    def test_read_and_aggregate_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.xlsx"
            make_xlsx(path)
            with XlsxReader(str(path)) as workbook:
                rows = list(workbook.rows_as_dicts("Report (2)"))
            orders = aggregate_work_orders(rows)
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(orders), 1)
            self.assertEqual(orders[0].work_order_id, "G1")
            self.assertEqual(len(orders[0].parts), 2)
            self.assertEqual(orders[0].dtc_codes, ["123"])


if __name__ == "__main__":
    unittest.main()

