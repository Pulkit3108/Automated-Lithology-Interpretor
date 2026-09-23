# Contributing

Keep changes focused on the public, local demonstration.

## Before submitting a change

1. Run `python -m unittest discover -s tests -v`.
2. Confirm that the application makes no external network calls.
3. Confirm that new fixtures are synthetic or have explicit redistribution terms.
4. Check that no credential, private URL, employee information, absolute workstation path, serialized model, generated cache, or office binary is included.
5. Document any new input column, unit assumption, preprocessing step, or evaluation limitation.

Do not add a live service integration or model deserialization path without a separate security review and an explicit repository decision.

