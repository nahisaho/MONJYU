# Community Report Generator - FEAT-013
"""
MONJYU Community Report Generator Module

コミュニティのエグゼクティブサマリーをLLMで生成
"""

from monjyu.index.community_report_generator.types import (
    Finding,
    CommunityReport,
    ReportGenerationResult,
)
from monjyu.index.community_report_generator.prompts import (
    get_prompts,
    build_report_prompt,
    format_entities_section,
    format_relationships_section,
)
from monjyu.index.community_report_generator.generator import (
    CommunityReportGenerator,
    CommunityReportGeneratorConfig,
    CommunityReportGeneratorProtocol,
    ChatModelProtocol,
    SyncLLMAdapter,
    create_report_generator,
)

__all__ = [
    # Types
    "Finding",
    "CommunityReport",
    "ReportGenerationResult",
    # Prompts
    "get_prompts",
    "build_report_prompt",
    "format_entities_section",
    "format_relationships_section",
    # Generator
    "CommunityReportGenerator",
    "CommunityReportGeneratorConfig",
    "CommunityReportGeneratorProtocol",
    "ChatModelProtocol",
    "SyncLLMAdapter",
    "ChatModelProtocol",
    "create_report_generator",
]
