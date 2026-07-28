import unittest

from examples.intent_routing.schema import create_intent_routing_schema


class IntentRoutingSchemaTests(unittest.TestCase):
    def test_valid_fixed_object(self):
        valid, errors = create_intent_routing_schema().validate({"intent": "delivery", "urgency": "high", "evidence": "三天了还没收到"})
        self.assertTrue(valid, errors)

    def test_invalid_enum_and_evidence_length_are_rejected(self):
        valid, _ = create_intent_routing_schema().validate({"intent": "price", "urgency": "urgent", "evidence": "x" * 41})
        self.assertFalse(valid)


if __name__ == "__main__":
    unittest.main()
