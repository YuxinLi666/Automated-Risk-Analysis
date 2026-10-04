import hashlib
import os
import unittest
from pathlib import Path

from engine_fault_nlp.ingest import audit_workbook


WORKBOOK = Path(os.environ.get("ENGINE_FAULT_TEST_WORKBOOK", "data/private/source_workbook.xlsx"))
EXPECTED_SHA256 = "A7B9479F0EC9F6C862149963FCDE2FBD2F8EDFB7B600CDC20197FB34C6841FE6"


@unittest.skipUnless(WORKBOOK.is_file(), "private historical workbook is not available")
class RealWorkbookContractTest(unittest.TestCase):
    def test_report_snapshot_and_source_hash(self):
        digest = hashlib.sha256(WORKBOOK.read_bytes()).hexdigest().upper()
        self.assertEqual(digest, EXPECTED_SHA256)
        audit = audit_workbook(str(WORKBOOK), "Report")
        self.assertEqual(audit["rows"], 3832)
        self.assertEqual(audit["rows_with_work_order_id"], 3826)
        self.assertEqual(audit["rows_without_work_order_id"], 6)
        self.assertEqual(audit["unique_work_orders"], 2638)
        self.assertEqual(audit["duplicate_work_order_groups"], 425)
        self.assertEqual(audit["max_rows_per_work_order"], 30)


if __name__ == "__main__":
    unittest.main()
