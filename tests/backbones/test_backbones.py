import torch


def test_genbiobert(genbiobert):
    model = genbiobert

    # Test forward method
    input_ids = torch.randint(0, model.get_vocab_size(), (4, 10))
    attention_mask = torch.ones(4, 10)
    output = model.forward(input_ids, attention_mask)
    assert output.shape == (4, 10, 16)

    # Test tokenize method
    sequences = ["ACGT", "TGC"]
    input_ids, attention_mask, special_mask = model.tokenize(
        sequences, padding=True, add_special_tokens=True
    )
    assert tuple(map(len, input_ids)) == (6, 6)
    assert tuple(map(len, attention_mask)) == (6, 6)
    assert tuple(map(len, special_mask)) == (6, 6)
    input_ids, attention_mask, special_mask = model.tokenize(
        sequences, padding=True, add_special_tokens=False
    )
    assert tuple(map(len, input_ids)) == (4, 4)
    assert tuple(map(len, attention_mask)) == (4, 4)
    assert tuple(map(len, special_mask)) == (4, 4)
    input_ids, attention_mask, special_mask = model.tokenize(
        sequences, padding=False, add_special_tokens=False
    )
    assert tuple(map(len, input_ids)) == (4, 3)
    assert tuple(map(len, attention_mask)) == (4, 3)
    assert tuple(map(len, special_mask)) == (4, 3)

    # Test decode_tokens method
    decoded_sequences = model.decode_tokens(input_ids)
    assert decoded_sequences == ["A C G T", "T G C"]

    # Test get_token_id method
    token_id = model.get_token_id("A")
    assert token_id == 5

    # Test get_max_context method
    max_context = model.get_max_context()
    assert max_context == 128

    # Test get_embedding_size method
    embedding_size = model.get_embedding_size()
    assert embedding_size == 16

    # Test get_vocab_size method
    vocab_size = model.get_vocab_size()
    assert vocab_size == 16

    # Test get_num_layer method
    num_layers = model.get_num_layer()
    assert num_layers == 2


def test_genbiofm(genbiofm):
    model = genbiofm

    # Test forward method
    input_ids = torch.randint(0, model.get_vocab_size(), (4, 10))
    attention_mask = torch.ones(4, 10)
    output = model.forward(input_ids, attention_mask)
    assert output.shape == (4, 10, 16)

    # Test tokenize method
    sequences = ["ACGT", "TGC"]
    input_ids, attention_mask, special_mask = model.tokenize(
        sequences, padding=True, add_special_tokens=True
    )
    assert tuple(map(len, input_ids)) == (5, 5)
    assert tuple(map(len, attention_mask)) == (5, 5)
    assert tuple(map(len, special_mask)) == (5, 5)
    input_ids, attention_mask, special_mask = model.tokenize(
        sequences, padding=True, add_special_tokens=False
    )
    assert tuple(map(len, input_ids)) == (4, 4)
    assert tuple(map(len, attention_mask)) == (4, 4)
    assert tuple(map(len, special_mask)) == (4, 4)
    input_ids, attention_mask, special_mask = model.tokenize(
        sequences, padding=False, add_special_tokens=False
    )
    assert tuple(map(len, input_ids)) == (4, 3)
    assert tuple(map(len, attention_mask)) == (4, 3)
    assert tuple(map(len, special_mask)) == (4, 3)

    # Test decode_tokens method
    decoded_sequences = model.decode_tokens(input_ids)
    assert decoded_sequences == ["A C G T", "T G C"]

    # Test get_token_id method
    token_id = model.get_token_id("A")
    assert token_id == 2

    # Test get_max_context method
    max_context = model.get_max_context()
    assert max_context == 128

    # Test get_embedding_size method
    embedding_size = model.get_embedding_size()
    assert embedding_size == 16

    # Test get_vocab_size method
    vocab_size = model.get_vocab_size()
    assert vocab_size == 128

    # Test get_num_layer method
    num_layers = model.get_num_layer()
    assert num_layers == 2


def test_genbiocellfoundation(genbiocellfoundation):
    model = genbiocellfoundation

    # Test forward method
    input_ids = torch.randint(0, 128, (4, 10))
    attention_mask = torch.ones(4, 10)
    output = model.forward(input_ids, attention_mask)
    assert output.shape == (4, 8, 16)  # Adjusted for trimmed embeddings

    # Test tokenize method
    sequences = torch.randint(0, 128, (4, 10))
    input_ids, attention_mask, special_mask = model.tokenize(
        sequences, padding=True, add_special_tokens=True
    )
    assert input_ids.shape == (4, 10)
    assert attention_mask is None
    assert special_mask is None

    # Test get_max_context method
    max_context = model.get_max_context()
    assert max_context == 128

    # Test get_embedding_size method
    embedding_size = model.get_embedding_size()
    assert embedding_size == 16

    # Test get_num_layer method
    num_layers = model.get_num_layer()
    assert num_layers == 2


def test_enformer(enformer):
    model = enformer

    # Test forward method
    input_ids = torch.randn(4, 256, 4)  # One-hot encoded input
    attention_mask = torch.ones(4, 256)
    output = model.forward(input_ids, attention_mask)
    assert output.shape == (4, 2, 24)

    # Test tokenize method
    sequences = ["ACGT" * 64, "TGCA" * 64]
    input_ids, attention_mask, special_mask = model.tokenize(
        sequences, padding=True, add_special_tokens=False
    )
    assert input_ids.shape == (2, 256, 4)
    assert attention_mask.shape == (2, 2)
    assert special_mask is None

    # Test get_max_context method
    max_context = model.get_max_context()
    assert max_context == 128

    # Test get_embedding_size method
    embedding_size = model.get_embedding_size()
    assert embedding_size == 24

    # Test get_num_layer method
    num_layers = model.get_num_layer()
    assert num_layers == 2


def test_esm(esm):
    model = esm
    # Test forward method
    input_ids = torch.randint(0, model.get_vocab_size(), (4, 10))
    attention_mask = torch.ones(4, 10)
    output = model.forward(input_ids, attention_mask)
    assert output.shape == (4, 10, 16)

    # Test tokenize method
    sequences = ["ACDEFGHIK", "LMNPQRSTVWY"]
    input_ids, attention_mask, special_mask = model.tokenize(sequences)
    assert tuple(map(len, input_ids)) == (13, 13)
    assert tuple(map(len, attention_mask)) == (13, 13)
    assert tuple(map(len, special_mask)) == (13, 13)

    # Test decode_tokens method
    decoded_sequences = model.decode_tokens(input_ids)
    assert decoded_sequences == [
        "<cls> A C D E F G H I K <eos> <pad> <pad>",
        "<cls> L M N P Q R S T V W Y <eos>",
    ]

    # Test get_token_id method
    token_id = model.get_token_id("A")
    assert token_id == 5

    # Test get_max_context method
    max_context = model.get_max_context()
    assert max_context == 128

    # Test get_embedding_size method
    embedding_size = model.get_embedding_size()
    assert embedding_size == 16

    # Test get_vocab_size method
    vocab_size = model.get_vocab_size()
    assert vocab_size == 33
