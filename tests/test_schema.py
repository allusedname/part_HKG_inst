from partcat_hkg.data.schema import RoleSchema


def test_schema_smoke():
    schema = RoleSchema.from_names(["car"], ["body", "wheel"], ["car:body", "car:wheel"])
    assert schema.smoke_test() == {"invalid_role_mappings": 0, "duplicate_role_slots": 0}
    assert schema.role_for(0, 1) == 1
