def test_parse_a0():
    line = "EQUILIBRIUM_A0 3.18641752884217"
    # Fails: float(line.split("=")[-1].strip())
    val = float(line.split()[1])
    assert val == 3.18641752884217

test_parse_a0()
print("Pass")
