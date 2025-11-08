import torch


def test_genbiobert(genbiobert):
    model = genbiobert

    # Test forward method
    input_ids = torch.randint(0, model.get_vocab_size(), (4, 10))
    attention_mask = torch.ones(4, 10)
    output = model.forward(input_ids, attention_mask)
    assert output.last_hidden_state.shape == (4, 10, 16)

    # Test tokenize method
    sequences = ["ACGT", "TGC"]
    processed_batch = model.process_batch({"sequences": sequences}, None)
    input_ids = processed_batch["input_ids"]
    attention_mask = processed_batch["attention_mask"]
    special_mask = processed_batch["special_tokens_mask"]
    assert input_ids.shape == (2, 6)
    assert attention_mask.shape == (2, 6)
    assert special_mask.shape == (2, 6)
    processed_batch = model.process_batch({"sequences": sequences}, None, add_special_tokens=False)
    input_ids = processed_batch["input_ids"]
    attention_mask = processed_batch["attention_mask"]
    special_mask = processed_batch["special_tokens_mask"]
    assert input_ids.shape == (2, 4)
    assert attention_mask.shape == (2, 4)
    assert special_mask.shape == (2, 4)

    # Test decode_tokens method
    decoded_sequences = model.decode_tokens(input_ids)
    assert decoded_sequences == ["A C G T", "T G C [PAD]"]

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
    assert output.last_hidden_state.shape == (4, 10, 16)

    # Test tokenize method
    sequences = ["ACGT", "TGC"]
    processed_batch = model.process_batch({"sequences": sequences}, None)
    input_ids = processed_batch["input_ids"]
    attention_mask = processed_batch["attention_mask"]
    special_mask = processed_batch["special_tokens_mask"]
    assert input_ids.shape == (2, 5)
    assert attention_mask.shape == (2, 5)
    assert special_mask.shape == (2, 5)
    processed_batch = model.process_batch({"sequences": sequences}, None, add_special_tokens=False)
    input_ids = processed_batch["input_ids"]
    attention_mask = processed_batch["attention_mask"]
    special_mask = processed_batch["special_tokens_mask"]
    assert input_ids.shape == (2, 4)
    assert attention_mask.shape == (2, 4)
    assert special_mask.shape == (2, 4)

    # Test decode_tokens method
    decoded_sequences = model.decode_tokens(input_ids)
    assert decoded_sequences == ["A C G T", "T G C [PAD]"]

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
    assert output.last_hidden_state.shape == (4, 8, 16)

    # Test process_batch method
    sequences = torch.randint(0, 128, (4, 8), device=device, dtype=torch.bfloat16)
    processed_batch = model.process_batch({"sequences": sequences}, None)
    assert sequences.equal(processed_batch["input_ids"])
    assert "attention_mask" not in processed_batch
    assert "special_tokens_mask" not in processed_batch

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
    embeds = output.last_hidden_state
    assert embeds.shape[0] == 4 and embeds.shape[2] == embedding_size
    assert output.attention_mask is None
    # only nonzero input into model, so the dim value depends on random generation input.
    # Two cells
    input_ids = torch.randint(0, 128, (4, max_context * 2), device=device, dtype=torch.bfloat16)
    output = model.forward(input_ids, None)
    embeds = output.last_hidden_state
    attention_mask = output.attention_mask
    assert embeds.shape[0] == 4 and embeds.shape[2] == embedding_size
    assert attention_mask.shape == embeds.shape[:2]
    assert (input_ids[0, :max_context] > 0).sum() + 2 == (attention_mask[0, :] > 0).sum()


def test_enformer(enformer):
    model = enformer

    # Test forward method
    input_ids = torch.randn(4, 256, 4)  # One-hot encoded input
    assert model.forward(input_ids).last_hidden_state.shape == (4, 2, 24)

    # Test tokenize method
    sequences = ["ACGT" * 64, "TGCA" * 64]
    processed_batch = model.process_batch({"sequences": sequences}, None)
    assert processed_batch["input_ids"].shape == (2, 256, 4)
    assert "attention_mask" not in processed_batch
    assert "special_tokens_mask" not in processed_batch

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
    output = model.forward(input_ids)
    embeds = output.last_hidden_state
    attention_mask = output.attention_mask
    assert embeds.shape == (5, 2, model.get_embedding_size())
    assert attention_mask.shape == embeds.shape[:2]

    # Test tokenize method
    sequences = ["ACGT" * 64, "TGCA" * 64]
    processed_batch = model.process_batch({"sequences": sequences}, None)
    input_ids = processed_batch["input_ids"]
    assert input_ids.shape == (2, 4, 256)
    assert "attention_mask" not in processed_batch
    assert "special_tokens_mask" not in processed_batch

    # Test get_max_context method
    max_context = model.get_max_context()
    assert max_context == 128

    # Test get_embedding_size method
    embedding_size = model.get_embedding_size()
    # Borzoi is hard-coded to 1920 output dim always
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
    assert output.last_hidden_state.shape == (4, 10, 16)
    assert output.attention_mask.shape == (4, 10)

    # Test process_batch method
    sequences = ["ACDEFGHIK", "LMNPQRSTVWY"]
    tokenized_result = model.tokenize(sequences)
    input_ids = tokenized_result["input_ids"]
    attention_mask = tokenized_result["attention_mask"]
    special_mask = tokenized_result["special_tokens_mask"]
    assert tuple(map(len, input_ids)) == (13, 13)
    assert tuple(map(len, attention_mask)) == (13, 13)
    assert tuple(map(len, special_mask)) == (13, 13)

    # Test decode_tokens method
    decoded_sequences = model.decode_tokens(input_ids)
    assert decoded_sequences == [
        "<cls> A C D E F G H I K <eos> <pad> <pad>",
        "<cls> L M N P Q R S T V W Y <eos>",
    ]

    # Test process_batch method
    processed_batch = model.process_batch({"sequences": sequences}, None)
    assert processed_batch["input_ids"].shape == (2, 13)
    assert processed_batch["attention_mask"].shape == (2, 13)
    assert processed_batch["special_tokens_mask"].shape == (2, 13)

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


def test_geneformer_forward_and_getters(geneformer):
    model = geneformer

    # feed random token IDs and an all-ones mask
    input_ids = torch.randint(0, 100, (2, 10), dtype=torch.long)
    attention_mask = torch.ones(2, 10, dtype=torch.long)

    out = model.forward(input_ids, attention_mask).last_hidden_state
    # expect (batch, seq_len, hidden_size)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 10, 16)

    # getters
    assert model.get_max_context() == 10
    assert model.get_embedding_size() == 16
    # default has no decoder
    assert model.get_decoder() is None


def test_geneformer_decoder_for_masked_lm_head(geneformer_mlm):
    model = geneformer_mlm
    dec = model.get_decoder()

    fake_hidden = torch.randn(2, 5, 8)
    logits = dec(fake_hidden)
    assert isinstance(logits, torch.Tensor)
    # logits shape should be (batch, seq_len, vocab_size)
    assert logits.ndim == 3


def test_scimilarity_forward(scimilarity):
    """Forward pass returns an embedding of expected shape."""
    batch_size = 3
    dummy_counts = torch.rand(batch_size, scimilarity.num_genes)
    batch = {"sequences": dummy_counts}

    outputs = scimilarity(**scimilarity.process_batch(batch, device=torch.device("cpu")))
    assert outputs.last_hidden_state.shape == (
        batch_size,
        scimilarity.latent_dim,
    )


def test_scimilarity_embedding_size(scimilarity):
    """`get_embedding_size` should match the encoder latent dimension."""
    assert scimilarity.get_embedding_size() == scimilarity.latent_dim
