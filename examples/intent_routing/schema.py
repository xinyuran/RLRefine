"""A second, independent Schema example: support-ticket intent routing."""
from core.schema import FieldDefinition, FieldType, TaskSchema


INTENTS = ["refund", "delivery", "product_quality", "account", "other"]
URGENCY = ["low", "normal", "high"]


def create_intent_routing_schema() -> TaskSchema:
    schema = TaskSchema(
        name="support_ticket_routing",
        description="Route a customer-support ticket using an evidence span copied from the source.",
    )
    schema.add_field(FieldDefinition(name="intent", type=FieldType.STRING, enum_values=INTENTS))
    schema.add_field(FieldDefinition(name="urgency", type=FieldType.STRING, enum_values=URGENCY))
    schema.add_field(FieldDefinition(name="evidence", type=FieldType.STRING, min_length=1, max_length=40))
    return schema
