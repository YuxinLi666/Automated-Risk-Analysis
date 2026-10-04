import unittest

from engine_fault_nlp.cause_extraction import extract_diagnostic_cause


class CauseExtractionTest(unittest.TestCase):
    def test_short_component_failure(self):
        result = extract_diagnostic_cause("温度传感器功能失效", "温度传感器")
        self.assertEqual(result.text, "温度传感器功能失效")
        self.assertGreater(result.confidence, 0.8)

    def test_causal_phrase(self):
        result = extract_diagnostic_cause("由于点火线圈损坏所致故障", "点火线圈")
        self.assertEqual(result.text, "点火线圈损坏")
        self.assertEqual(result.method, "causal_pattern_2")

    def test_symptom_only_is_not_cause(self):
        result = extract_diagnostic_cause("发动机限扭")
        self.assertEqual(result.text, "")

    def test_component_fault_is_preserved(self):
        result = extract_diagnostic_cause("MTM选挡机构故障")
        self.assertEqual(result.text, "MTM选挡机构故障")

    def test_no_specific_cause_does_not_match_reason_substring(self):
        result = extract_diagnostic_cause("现场3天未查找到具体原因,与厂内工程师返回协商进一步处理措施。")
        self.assertEqual(result.text, "")

    def test_diagnostic_narrative_uses_judgement(self):
        text = "检查蓄电池电压正常，启动时起动机呜呜响但发动机不转动。检查飞轮齿圈正常，判断为起动机单向器打滑；更换后正常。"
        result = extract_diagnostic_cause(text, "起动机")
        self.assertEqual(result.text, "起动机单向器打滑")


if __name__ == "__main__":
    unittest.main()
