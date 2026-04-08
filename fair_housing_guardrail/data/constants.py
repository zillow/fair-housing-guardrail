# Multi-class mapping
MULTICLASS_ID_TO_LABEL = {
    0: "compliant",
    1: "non-compliant-age",
    2: "non-compliant-disability",
    3: "non-compliant-familial/marital status",
    4: "non-compliant-housing assistance",
    5: "non-compliant-offensive",
    6: "non-compliant-politics",
    7: "non-compliant-race/color/ethnicity/national origin",
    8: "non-compliant-religion",
    9: "non-compliant-sex/gender identity/sexual orientation",
    10: "non-compliant-veteran status",
}
MULTICLASS_LABEL_TO_ID = {v: k for k, v in MULTICLASS_ID_TO_LABEL.items()}

# Binary mapping
BINARY_ID_TO_LABEL = {1: "compliant", 0: "non-compliant"}
BINARY_LABEL_TO_ID = {v: k for k, v in BINARY_ID_TO_LABEL.items()}


# Helper to select mapping based on label set
def get_label_mappings(labels):
    """
    Given a set of labels, return the appropriate (id_to_label, label_to_id) mapping.
    """
    if set(labels) <= set(BINARY_LABEL_TO_ID.keys()):
        return BINARY_ID_TO_LABEL, BINARY_LABEL_TO_ID
    return MULTICLASS_ID_TO_LABEL, MULTICLASS_LABEL_TO_ID
