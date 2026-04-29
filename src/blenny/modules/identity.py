"""A no-op pipeline step. Useful as a template and for tests."""

from __future__ import annotations

from blenny.pipeline.core import Context, PipelineStep


class IdentityPreprocessor(PipelineStep):
    """Returns the context unchanged. The 'hello world' of pipeline steps."""

    name = "IdentityPreprocessor"

    def run(self, context: Context) -> Context:
        return context
