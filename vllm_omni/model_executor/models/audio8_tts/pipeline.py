# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio8 TTS pipeline topology.

Stage 0: audio8_tts_ar — text (+ optional voice-clone conditioning) → 10-
         codebook frame stacks via the dual-AR slow backbone + fast head.
Stage 1: audio8_codec — code stacks → waveform @44.1 kHz (ArkttsCodec).

The HF checkpoint reports ``model_type = "arktts"``; that string is the
registry key so bare ``vllm-omni serve <repo>`` auto-detects.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.audio8_tts"

AUDIO8_TTS_PIPELINE = PipelineConfig(
    model_type="arktts",
    default_deploy_config_name="audio8_tts.yaml",
    # The checkpoint config.json architectures entry.
    model_arch="ArkttsModel",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="audio8_tts_ar",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=f"{_PROC}.ar2codec_async_chunk",
            sampling_constraints={
                "detokenize": False,
                # <|im_end|> — stop when the AR model emits end-of-turn.
                "stop_token_ids": [151645],
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="audio8_codec",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="Audio8CodecDecoder",
            sync_process_input_func=f"{_PROC}.ar2codec",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
