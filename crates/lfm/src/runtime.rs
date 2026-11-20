use anyhow::Result;
use lfm_app::InferenceBackend;
use lfm_chat::ChatTemplate;
use lfm_gpu::{GpuModel, PipelineModel, TpModel};
use lfm_tokenizer::LfmTokenizer;

use crate::config::{BackendMode, Config};

pub struct Runtime {
    pub backend: Box<dyn InferenceBackend>,
    pub tokenizer: LfmTokenizer,
    pub template: ChatTemplate,
    pub max_ctx: usize,
}

impl Runtime {
    pub fn open(cfg: &Config) -> Result<Self> {
        cfg.validate()?;
        let opts = cfg.backend.to_options();
        let ctx = cfg.backend.ctx;
        let backend: Box<dyn InferenceBackend> = match cfg.backend.mode {
            BackendMode::Single => Box::new(GpuModel::load_with(&cfg.model.path, ctx, &opts)?),
            BackendMode::Pipeline => {
                let tiers: Vec<(usize, usize)> = cfg
                    .backend
                    .devices
                    .iter()
                    .copied()
                    .zip(cfg.backend.effective_tiers()?)
                    .collect();
                Box::new(PipelineModel::load_tiers_with(
                    &cfg.model.path,
                    ctx,
                    &tiers,
                    &opts,
                )?)
            }
            BackendMode::Tp => Box::new(TpModel::load_with(
                &cfg.model.path,
                ctx,
                &cfg.backend.devices,
                &opts,
            )?),
        };
        let tokenizer = LfmTokenizer::from_file(&cfg.model.tokenizer)?;
        let template = ChatTemplate::from_file(&cfg.model.template)?;
        let max_ctx = backend.max_ctx();
        Ok(Self {
            backend,
            tokenizer,
            template,
            max_ctx,
        })
    }
}
