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


def test_genbiocellfoundation(genbiocellfoundation, flash_attn_available):
    model = genbiocellfoundation
    device = torch.device("cuda:0" if flash_attn_available else "cpu")
    model = model.to(device, torch.bfloat16)
    # Test forward method
    input_ids = torch.randint(0, 128, (4, 8), device=device, dtype=torch.bfloat16)
    output = model.forward(input_ids, None)
    assert output.shape == (4, 8, 16)

    # Test tokenize method
    sequences = torch.randint(0, 128, (4, 8), device=device, dtype=torch.bfloat16)
    input_ids, attention_mask, special_mask = model.tokenize(
        sequences, padding=True, add_special_tokens=True
    )
    assert sequences.equal(input_ids)
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


def test_genbiocellspatialfoundation(genbiocellspatialfoundation, flash_attn_available):
    model = genbiocellspatialfoundation
    device = torch.device("cuda:0" if flash_attn_available else "cpu")
    model = model.to(device, torch.bfloat16)

    # Test get_max_context method
    max_context = model.get_max_context()
    assert max_context == 128

    # Test get_embedding_size method
    embedding_size = model.get_embedding_size()
    assert embedding_size == 16

    # Test get_num_layer method
    num_layers = model.get_num_layer()
    assert num_layers == 2

    # Test forward method
    # Single cell
    input_ids = torch.randint(0, 128, (4, max_context), device=device, dtype=torch.bfloat16)
    output = model.forward(input_ids, None)
    assert output.shape[0] == 4 and output.shape[2] == embedding_size
    # only nonzero input into model, so the dim value depends on random generation input.
    # Two cells
    input_ids = torch.randint(0, 128, (4, max_context * 2), device=device, dtype=torch.bfloat16)
    output = model.forward(input_ids, None)
    assert output.shape[0] == 4 and output.shape[2] == embedding_size

    # Test tokenize method
    # Single cell
    sequences = torch.randint(0, 128, (4, max_context), device=device, dtype=torch.bfloat16)
    input_ids, attention_mask, special_mask = model.tokenize(sequences)
    assert sequences.equal(input_ids)
    assert attention_mask.shape[0] == 4
    assert special_mask is None
    # check for nonzero positions in input_ids & attention_mask
    assert (input_ids[0, :] > 0).sum() + 2 == (attention_mask[0, :] > 0).sum()
    # Two cells
    sequences = torch.randint(0, 128, (4, max_context * 2), device=device, dtype=torch.bfloat16)
    input_ids, attention_mask, special_mask = model.tokenize(sequences)
    assert sequences.equal(input_ids)
    assert attention_mask.shape[0] == 4
    assert special_mask is None
    # attention_mask is 1st cell padding mask, not all cells
    assert (input_ids[0, :max_context] > 0).sum() + 2 == (attention_mask[0, :] > 0).sum()


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
    assert num_layers == 1


def test_borzoi(borzoi):
    model = borzoi

    # Test forward method
    input_ids = torch.randn(5, 4, 256)  # One-hot encoded input
    attention_mask = torch.ones(5, 256)
    output = model.forward(input_ids, attention_mask)
    assert output.shape == (5, 2, 1920)  # Borzoi is hard-coded to 1920 output dim always
    assert output.shape[-1] == model.get_embedding_size()

    # Test tokenize method
    sequences = ["ACGT" * 64, "TGCA" * 64]
    input_ids, attention_mask, special_mask = model.tokenize(
        sequences, padding=True, add_special_tokens=False
    )
    assert input_ids.shape == (2, 4, 256)
    assert attention_mask.shape == (2, 2)
    assert special_mask is None

    # Test get_max_context method
    max_context = model.get_max_context()
    assert max_context == 128

    # Test get_embedding_size method
    embedding_size = model.get_embedding_size()
    assert embedding_size == 1920

    # Test get_num_layer method
    num_layers = model.get_num_layer()
    assert num_layers == 1


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
