from hermes.cli import is_goodbye


def test_goodbye_variants():
    assert is_goodbye("adios")
    assert is_goodbye("adiós")
    assert is_goodbye("Adios!")
    assert is_goodbye("hasta luego")
    assert is_goodbye("bye")
    assert is_goodbye("chao")
    assert is_goodbye("nos vemos.")


def test_not_goodbye():
    assert not is_goodbye("adios a todos")
    assert not is_goodbye("dime un chiste")
    assert not is_goodbye("")
    assert not is_goodbye("hola")
