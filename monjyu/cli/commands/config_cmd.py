# MONJYU CLI - Config Commands
"""
FEAT-008: CLI - config コマンド群
設定ファイルの生成・表示・検証
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.table import Table

from monjyu.cli.main import print_error, print_success, print_warning

console = Console()
config_app = typer.Typer(help="Configuration commands")


# デフォルト設定テンプレート
DEFAULT_CONFIG = """# MONJYU Configuration File
# Academic Paper RAG System using Progressive GraphRAG

# ======================================
# Basic Settings
# ======================================

# Output directory for index data
output_path: ./output

# Environment: local (Ollama) or azure (Azure OpenAI)
environment: local

# Index levels to build (0: basic, 1: advanced with communities)
index_levels: [0, 1]

# ======================================
# Search Settings
# ======================================

# Default search mode: vector, lazy, or auto
default_search_mode: lazy

# Default number of search results
default_top_k: 10

# ======================================
# Document Processing
# ======================================

# Text chunk size (tokens)
chunk_size: 1200

# Overlap between chunks (tokens)
chunk_overlap: 100

# ======================================
# Local Environment (Ollama)
# ======================================

# LLM model for text generation
llm_model: llama3:8b-instruct-q4_K_M

# Embedding model
embedding_model: nomic-embed-text

# Ollama API endpoint
ollama_base_url: http://192.168.224.1:11434

# ======================================
# Azure Environment (Optional)
# ======================================

# Uncomment and configure for Azure deployment
# azure_openai_endpoint: https://your-resource.openai.azure.com/
# azure_openai_api_key: ${AZURE_OPENAI_API_KEY}
# azure_openai_deployment: gpt-4o
# azure_embedding_deployment: text-embedding-3-small

# Azure AI Search (Optional)
# azure_search_endpoint: https://your-search.search.windows.net/
# azure_search_api_key: ${AZURE_SEARCH_API_KEY}
"""


@config_app.command("init")
def config_init(
    output_path: Path = typer.Option(
        Path("./monjyu.yaml"),
        "--output", "-o",
        help="Output path for config file",
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing file"
    ),
):
    """Initialize a new configuration file"""
    
    if output_path.exists() and not force:
        print_warning(f"Config file already exists: {output_path}")
        console.print("Use --force to overwrite")
        raise typer.Exit(1)
    
    try:
        # 親ディレクトリを作成
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(DEFAULT_CONFIG)
        
        print_success(f"Configuration file created: {output_path}")
        console.print("\nEdit the file to customize your settings:")
        console.print(f"  [cyan]$EDITOR {output_path}[/cyan]")
        console.print("\nThen build your index:")
        console.print("  [cyan]monjyu index build ./papers/[/cyan]")
    
    except Exception as e:
        print_error(f"Failed to create config file: {e}")
        raise typer.Exit(1)


@config_app.command("show")
def config_show(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
):
    """Show current configuration"""
    
    # 設定ファイルを探す
    config_path = config
    if config_path is None:
        for candidate in [
            Path("./monjyu.yaml"),
            Path("./monjyu.yml"),
            Path("./config/monjyu.yaml"),
        ]:
            if candidate.exists():
                config_path = candidate
                break
    
    if config_path is None or not config_path.exists():
        print_warning("No configuration file found")
        console.print("\nCreate one with:")
        console.print("  [cyan]monjyu config init[/cyan]")
        raise typer.Exit(1)
    
    try:
        with open(config_path, encoding="utf-8") as f:
            content = f.read()
        
        syntax = Syntax(content, "yaml", theme="monokai", line_numbers=True)
        console.print(Panel(
            syntax,
            title=str(config_path),
            border_style="cyan",
        ))
    
    except Exception as e:
        print_error(f"Failed to read config file: {e}")
        raise typer.Exit(1)


@config_app.command("validate")
def config_validate(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
):
    """Validate configuration file"""
    
    # 設定ファイルを探す
    config_path = config or Path("./monjyu.yaml")
    
    if not config_path.exists():
        print_error(f"Config file not found: {config_path}")
        raise typer.Exit(1)
    
    try:
        from monjyu.api import MONJYU
        
        monjyu = MONJYU(config_path)
        cfg = monjyu.config
        
        # 検証結果テーブル
        table = Table(title="Configuration Validation")
        table.add_column("Setting", style="cyan")
        table.add_column("Value")
        table.add_column("Status")
        
        # 各設定項目をチェック
        checks = [
            ("Environment", cfg.environment, "✓"),
            ("Output Path", str(cfg.output_path), "✓"),
            ("Index Levels", str([l.value for l in cfg.index_levels]), "✓"),
            ("Search Mode", cfg.default_search_mode.value, "✓"),
            ("Top K", str(cfg.default_top_k), "✓"),
            ("Chunk Size", str(cfg.chunk_size), "✓"),
        ]
        
        for name, value, status in checks:
            table.add_row(name, value, f"[green]{status}[/green]")
        
        console.print(table)
        print_success("Configuration is valid")
        
        # 追加情報
        if cfg.environment == "local":
            console.print("\n[dim]Using local environment (Ollama)[/dim]")
            console.print(f"  LLM: {cfg.llm_model}")
            console.print(f"  Embedding: {cfg.embedding_model}")
        elif cfg.environment == "azure":
            console.print("\n[dim]Using Azure environment[/dim]")
    
    except Exception as e:
        print_error(f"Configuration error: {e}")
        raise typer.Exit(1)


@config_app.command("env")
def config_env():
    """Show environment variables used by MONJYU"""
    
    import os
    
    env_vars = [
        ("AZURE_OPENAI_ENDPOINT", "Azure OpenAI endpoint URL"),
        ("AZURE_OPENAI_API_KEY", "Azure OpenAI API key"),
        ("AZURE_SEARCH_ENDPOINT", "Azure AI Search endpoint URL"),
        ("AZURE_SEARCH_API_KEY", "Azure AI Search API key"),
        ("OLLAMA_BASE_URL", "Ollama API endpoint"),
        ("MONJYU_CONFIG", "Default config file path"),
    ]
    
    table = Table(title="Environment Variables")
    table.add_column("Variable", style="cyan")
    table.add_column("Description")
    table.add_column("Status")
    
    for var, desc in env_vars:
        value = os.environ.get(var)
        if value:
            # マスク処理（APIキーなど）
            if "KEY" in var or "SECRET" in var:
                display = f"[green]Set[/green] ({value[:8]}...)"
            else:
                display = f"[green]{value}[/green]"
        else:
            display = "[dim]Not set[/dim]"
        
        table.add_row(var, desc, display)
    
    console.print(table)
