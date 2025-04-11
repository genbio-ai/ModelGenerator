from dataclasses import dataclass, field


@dataclass
class NodeEmbeddingConfig:
    n_node_type: int = 200
    n_chain_type: int = 100  # at most 100 chains
    d: int = 128
    dropout: float = 0.1
    mask_node_type: bool = True


@dataclass
class EdgeTypeEmbeddingConfig:
    max_res_offset: int = 32
    max_sym_offset: int = 2
    d: int = 128
    dropout: float = 0.1


@dataclass
class BesselsConfig:
    bessel_const: float = 40.0
    d: int = 128


@dataclass
class EdgeEmbeddingConfig:
    d: int = 128
    k_for_knn: int | None = 30
    edge_type_emb: EdgeTypeEmbeddingConfig = field(default_factory=EdgeTypeEmbeddingConfig)
    bessels: BesselsConfig = field(default_factory=BesselsConfig)


@dataclass
class EquiformerConfig:
    d: int = 128
    n_head: int = 4
    d_mult: int = 3
    dropout: float = 0.1
    attn_dropout: float = 0


@dataclass
class QuantizeConfig:
    dim: int = 384
    n_embed: int = 512
    normalize: bool = True
    decay: float = 0.99
    eps: float = 1e-05
    usage_threshold: float = 1e-09
    restart: int = 100


@dataclass
class EquiformerEncoderConfig:
    n_eqnet: int = 12
    node_emb: NodeEmbeddingConfig = field(default_factory=NodeEmbeddingConfig)
    edge_emb: EdgeEmbeddingConfig = field(default_factory=EdgeEmbeddingConfig)
    eqnet: EquiformerConfig = field(default_factory=EquiformerConfig)
    quantize: QuantizeConfig = field(default_factory=QuantizeConfig)


@dataclass
class StructureModuleConfig:
    c_s: int = 384
    c_z: int = 128
    c_ipa: int = 16
    c_resnet: int = 128
    no_heads_ipa: int = 12
    no_qk_points: int = 4
    no_v_points: int = 8
    dropout_rate: float = 0.1
    no_blocks: int = 8
    no_transition_layers: int = 1
    no_resnet_blocks: int = 2
    no_angles: int = 7
    trans_scale_factor: int = 10
    epsilon: float = 1e-8
    inf: float = 1e5


@dataclass
class FoldingTrunkConfig:
    num_blocks: int = 32
    sequence_state_dim: int = 768
    pairwise_state_dim: int = 128
    sequence_head_width: int = 32
    pairwise_head_width: int = 32
    position_bins: int = 32
    dropout: float = 0.1
    layer_drop: float = 0
    cpu_grad_checkpoint: bool = False

    max_recycles: int = 1
    chunk_size: int | None = None

    structure_module: StructureModuleConfig = field(default_factory=StructureModuleConfig)


@dataclass
class ESMFoldDecoderConfig:
    quantize_dim: int = 384
    lddt_head_hidden_dim: int = 128
    folding_trunk: FoldingTrunkConfig = field(default_factory=FoldingTrunkConfig)


@dataclass
class StructureTokenizerConfig:
    encoder_config: EquiformerEncoderConfig = field(
        default_factory=EquiformerEncoderConfig
    )
    decoder_config: ESMFoldDecoderConfig = field(default_factory=ESMFoldDecoderConfig)
    frozen_codebook: bool = False
