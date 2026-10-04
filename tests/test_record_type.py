import unittest

from engine_fault_nlp.record_type import classify_record_type
from engine_fault_nlp.schemas import WorkOrder


class RecordTypeTest(unittest.TestCase):
    def test_proactive_upgrade_is_maintenance(self):
        order = WorkOrder("M1", fault_name="15N发动机泄压阀主动升级", diagnosis="主动升级")
        self.assertEqual(classify_record_type(order), "MAINTENANCE")

    def test_engine_maintenance_is_not_fault(self):
        order = WorkOrder("M2", fault_name="发动机保养", diagnosis="赠送机油机滤")
        self.assertEqual(classify_record_type(order), "MAINTENANCE")

    def test_upgrade_with_explicit_damage_remains_fault(self):
        order = WorkOrder("F1", diagnosis="传感器损坏，更换后升级标定")
        self.assertEqual(classify_record_type(order), "FAULT")

    def test_normal_failure_is_fault(self):
        order = WorkOrder("F2", diagnosis="点火线圈损坏")
        self.assertEqual(classify_record_type(order), "FAULT")

    def test_engineering_test_is_not_fault(self):
        order = WorkOrder("T1", fault_name="协助西康工程师进行热平衡测试")
        self.assertEqual(classify_record_type(order), "TEST_SERVICE")

    def test_non_engine_reason_is_out_of_scope(self):
        order = WorkOrder("O1", diagnosis="车辆气瓶故障导致,非发动机原因")
        self.assertEqual(classify_record_type(order), "OUT_OF_SCOPE")


if __name__ == "__main__":
    unittest.main()
