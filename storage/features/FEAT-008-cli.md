# FEAT-008: CLI (Command Line Interface)

**フィーチャーID**: FEAT-008  
**名称**: コマンドラインインターフェース  
**フェーズ**: Phase 1 (MVP)  
**優先度**: P0 (必須)  
**ステータス**: Draft

---

## 1. 概要

MONJYUの全機能をコマンドラインから利用可能にするCLIツール。

### 1.1 スコープ

```bash
monjyu index ./papers/
monjyu search "What is Transformer?"
monjyu status
```

- **入力**: コマンドライン引数
- **処理**: Python APIの呼び出し
- **出力**: 標準出力 (テキスト/JSON/Rich)
- **特徴**: typer + rich による美しいCLI

### 1.2 関連要件

| 要件ID | 要件名 | 優先度 |
|--------|--------|--------|
| FR-EXT-CLI-001 | index コマンド | P0 |
| FR-EXT-CLI-002 | search コマンド | P0 |
| FR-EXT-CLI-003 | status コマンド | P0 |
| FR-EXT-CLI-004 | config コマンド | P1 |
| FR-EXT-CLI-005 | document コマンド | P1 |
| FR-EXT-CLI-006 | citation コマンド | P1 |

### 1.3 依存関係

- **依存**: FEAT-007 (Python API)
- **被依存**: なし

---

## 2. アーキテクチャ

### 2.1 コマンド構造

```
monjyu
├── index                 # インデックス構築
│   ├── build            # インデックスを構築
│   └── status           # インデックス状態を表示
├── search               # 検索
│   ├── query            # 検索クエリ実行（デフォルト）
│   └── interactive      # 対話モード
├── document             # ドキュメント操作
│   ├── list             # 一覧表示
│   ├── show             # 詳細表示
│   └── export           # エクスポート
├── citation             # 引用ネットワーク
│   ├── chain            # 引用チェーン表示
│   ├── related          # 関連論文表示
│   └── top              # 影響力ランキング
├── config               # 設定
│   ├── show             # 設定表示
│   ├── init             # 設定ファイル生成
│   └── validate         # 設定検証
└── version              # バージョン表示
```

### 2.2 コンポーネント図

```
┌─────────────────────────────────────────────────────────────────────┐
│                           MONJYU CLI                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                      Typer Application                          │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │ │
│  │  │ index    │ │ search   │ │ document │ │ citation         │  │ │
│  │  │ commands │ │ commands │ │ commands │ │ commands         │  │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                    │                                 │
│                                    ▼                                 │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                      CLI Utilities                              │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │ │
│  │  │ OutputFormat │  │ ProgressBar  │  │ ErrorHandler         │ │ │
│  │  │ (text/json)  │  │ (rich)       │  │                      │ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                    │                                 │
│                                    ▼                                 │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                      MONJYU Python API                          │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 詳細設計

### 3.1 メインアプリケーション

```python
# monjyu/cli/main.py

import typer
from pathlib import Path
from typing import Optional
from enum import Enum

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from monjyu import MONJYU, SearchMode

# === アプリケーション初期化 ===

app = typer.Typer(
    name="monjyu",
    help="MONJYU - Academic Paper RAG System",
    add_completion=False
)
console = Console()

# === サブコマンドグループ ===

index_app = typer.Typer(help="Index management commands")
search_app = typer.Typer(help="Search commands")
document_app = typer.Typer(help="Document management commands")
citation_app = typer.Typer(help="Citation network commands")
config_app = typer.Typer(help="Configuration commands")

app.add_typer(index_app, name="index")
app.add_typer(search_app, name="search")
app.add_typer(document_app, name="document")
app.add_typer(citation_app, name="citation")
app.add_typer(config_app, name="config")

# === グローバルオプション ===

class OutputFormat(str, Enum):
    text = "text"
    json = "json"

def get_monjyu(config_path: Optional[Path] = None) -> MONJYU:
    """MONJYUインスタンスを取得"""
    config_path = config_path or Path("./monjyu.yaml")
    if config_path.exists():
        return MONJYU(config_path)
    return MONJYU()

# === バージョン ===

@app.command()
def version():
    """Show version information"""
    from monjyu import __version__
    console.print(f"MONJYU v{__version__}")
```

### 3.2 Index コマンド

```python
# monjyu/cli/commands/index.py

from pathlib import Path
from typing import Optional
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from ..main import index_app, console, get_monjyu, OutputFormat

@index_app.command("build")
def index_build(
    path: Path = typer.Argument(..., help="Path to documents directory"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file path"),
    levels: Optional[str] = typer.Option("0,1", "--levels", "-l", help="Index levels (comma-separated)"),
    rebuild: bool = typer.Option(False, "--rebuild", "-r", help="Rebuild existing index"),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", "-o", help="Output format")
):
    """Build index from documents"""
    
    monjyu = get_monjyu(config)
    
    # レベルをパース
    level_list = [int(l.strip()) for l in levels.split(",")]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("Building index...", total=100)
        
        try:
            # インデックス構築
            status = monjyu.index(
                path,
                levels=[IndexLevel(l) for l in level_list],
                rebuild=rebuild
            )
            
            progress.update(task, completed=100)
            
            if output == OutputFormat.json:
                import json
                console.print_json(json.dumps({
                    "status": status.index_status.value,
                    "documents": status.document_count,
                    "text_units": status.text_unit_count,
                    "communities": status.community_count
                }))
            else:
                console.print(Panel.fit(
                    f"[green]✓ Index built successfully[/green]\n\n"
                    f"Documents: {status.document_count}\n"
                    f"Text Units: {status.text_unit_count}\n"
                    f"Noun Phrases: {status.noun_phrase_count}\n"
                    f"Communities: {status.community_count}\n"
                    f"Citation Edges: {status.citation_edge_count}",
                    title="Index Status"
                ))
        
        except Exception as e:
            progress.stop()
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)


@index_app.command("status")
def index_status(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", "-o")
):
    """Show index status"""
    
    monjyu = get_monjyu(config)
    status = monjyu.get_status()
    
    if output == OutputFormat.json:
        import json
        console.print_json(json.dumps({
            "status": status.index_status.value,
            "levels_built": [l.value for l in status.index_levels_built],
            "documents": status.document_count,
            "text_units": status.text_unit_count,
            "noun_phrases": status.noun_phrase_count,
            "communities": status.community_count,
            "citation_edges": status.citation_edge_count,
            "last_error": status.last_error
        }))
    else:
        table = Table(title="MONJYU Index Status")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Status", status.index_status.value)
        table.add_row("Levels Built", str([l.value for l in status.index_levels_built]))
        table.add_row("Documents", str(status.document_count))
        table.add_row("Text Units", str(status.text_unit_count))
        table.add_row("Noun Phrases", str(status.noun_phrase_count))
        table.add_row("Communities", str(status.community_count))
        table.add_row("Citation Edges", str(status.citation_edge_count))
        
        if status.last_error:
            table.add_row("Last Error", f"[red]{status.last_error}[/red]")
        
        console.print(table)
```

### 3.3 Search コマンド

```python
# monjyu/cli/commands/search.py

from pathlib import Path
from typing import Optional
import typer
from rich.panel import Panel
from rich.markdown import Markdown

from ..main import search_app, console, get_monjyu, OutputFormat
from monjyu import SearchMode

@search_app.callback(invoke_without_command=True)
def search_default(
    ctx: typer.Context,
    query: Optional[str] = typer.Argument(None, help="Search query"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    mode: str = typer.Option("lazy", "--mode", "-m", help="Search mode: vector, lazy, auto"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results"),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", "-o")
):
    """Execute search query"""
    
    if ctx.invoked_subcommand is not None:
        return
    
    if query is None:
        console.print("[yellow]Please provide a search query[/yellow]")
        raise typer.Exit(1)
    
    monjyu = get_monjyu(config)
    
    search_mode = SearchMode(mode)
    
    with console.status("[bold green]Searching..."):
        result = monjyu.search(query, mode=search_mode, top_k=top_k)
    
    if output == OutputFormat.json:
        import json
        console.print_json(json.dumps({
            "query": result.query,
            "answer": result.answer,
            "citations": [
                {
                    "document_id": c.document_id,
                    "title": c.document_title,
                    "snippet": c.text_snippet
                }
                for c in result.citations
            ],
            "search_mode": result.search_mode.value,
            "search_level": result.search_level,
            "total_time_ms": result.total_time_ms,
            "llm_calls": result.llm_calls
        }))
    else:
        # 回答パネル
        console.print(Panel(
            Markdown(result.answer),
            title=f"[bold]Answer[/bold] (mode: {result.search_mode.value}, level: {result.search_level})",
            border_style="green"
        ))
        
        # 引用
        if result.citations:
            console.print("\n[bold]Citations:[/bold]")
            for i, citation in enumerate(result.citations, 1):
                console.print(f"  [{i}] {citation.document_title}")
                console.print(f"      [dim]{citation.text_snippet[:100]}...[/dim]")
        
        # メタデータ
        console.print(f"\n[dim]Time: {result.total_time_ms:.0f}ms | LLM calls: {result.llm_calls}[/dim]")


@search_app.command("interactive")
def search_interactive(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    mode: str = typer.Option("lazy", "--mode", "-m")
):
    """Interactive search mode"""
    
    monjyu = get_monjyu(config)
    search_mode = SearchMode(mode)
    
    console.print(Panel.fit(
        "[bold]MONJYU Interactive Search[/bold]\n"
        f"Mode: {search_mode.value}\n"
        "Type 'exit' or 'quit' to exit\n"
        "Type 'mode <vector|lazy|auto>' to change mode",
        border_style="blue"
    ))
    
    current_mode = search_mode
    
    while True:
        try:
            query = console.input("\n[bold cyan]Query>[/bold cyan] ")
            
            if query.lower() in ["exit", "quit", "q"]:
                console.print("[dim]Goodbye![/dim]")
                break
            
            if query.lower().startswith("mode "):
                new_mode = query.split(" ", 1)[1].strip()
                try:
                    current_mode = SearchMode(new_mode)
                    console.print(f"[green]Mode changed to: {current_mode.value}[/green]")
                except ValueError:
                    console.print(f"[red]Invalid mode: {new_mode}[/red]")
                continue
            
            if not query.strip():
                continue
            
            with console.status("[bold green]Searching..."):
                result = monjyu.search(query, mode=current_mode)
            
            console.print(Panel(
                Markdown(result.answer),
                title="Answer",
                border_style="green"
            ))
            
            if result.citations:
                console.print("[dim]Citations:[/dim]")
                for i, c in enumerate(result.citations[:3], 1):
                    console.print(f"  [{i}] {c.document_title}")
        
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Type 'exit' to quit.[/dim]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
```

### 3.4 Document コマンド

```python
# monjyu/cli/commands/document.py

from pathlib import Path
from typing import Optional
import typer
from rich.table import Table
from rich.panel import Panel

from ..main import document_app, console, get_monjyu, OutputFormat

@document_app.command("list")
def document_list(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    limit: int = typer.Option(20, "--limit", "-n"),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", "-o")
):
    """List documents in index"""
    
    monjyu = get_monjyu(config)
    documents = monjyu.list_documents(limit=limit)
    
    if output == OutputFormat.json:
        import json
        console.print_json(json.dumps([
            {
                "id": d.id,
                "title": d.title,
                "authors": d.authors,
                "year": d.year,
                "chunks": d.chunk_count
            }
            for d in documents
        ]))
    else:
        table = Table(title=f"Documents ({len(documents)} shown)")
        table.add_column("ID", style="dim", max_width=12)
        table.add_column("Title", max_width=50)
        table.add_column("Authors", max_width=30)
        table.add_column("Year", justify="center")
        table.add_column("Chunks", justify="right")
        
        for doc in documents:
            authors = ", ".join(doc.authors[:2])
            if len(doc.authors) > 2:
                authors += f" +{len(doc.authors) - 2}"
            
            table.add_row(
                doc.id[:12],
                doc.title[:50] + ("..." if len(doc.title) > 50 else ""),
                authors,
                str(doc.year) if doc.year else "-",
                str(doc.chunk_count)
            )
        
        console.print(table)


@document_app.command("show")
def document_show(
    document_id: str = typer.Argument(..., help="Document ID"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", "-o")
):
    """Show document details"""
    
    monjyu = get_monjyu(config)
    doc = monjyu.get_document(document_id)
    
    if doc is None:
        console.print(f"[red]Document not found: {document_id}[/red]")
        raise typer.Exit(1)
    
    if output == OutputFormat.json:
        import json
        console.print_json(json.dumps({
            "id": doc.id,
            "title": doc.title,
            "authors": doc.authors,
            "year": doc.year,
            "doi": doc.doi,
            "chunk_count": doc.chunk_count,
            "citation_count": doc.citation_count,
            "reference_count": doc.reference_count,
            "influence_score": doc.influence_score
        }))
    else:
        console.print(Panel(
            f"[bold]{doc.title}[/bold]\n\n"
            f"ID: {doc.id}\n"
            f"Authors: {', '.join(doc.authors)}\n"
            f"Year: {doc.year or 'Unknown'}\n"
            f"DOI: {doc.doi or 'None'}\n"
            f"\n[bold]Index Stats:[/bold]\n"
            f"Chunks: {doc.chunk_count}\n"
            f"\n[bold]Citation Metrics:[/bold]\n"
            f"Citations: {doc.citation_count}\n"
            f"References: {doc.reference_count}\n"
            f"Influence Score: {doc.influence_score:.4f}",
            title="Document Details"
        ))


@document_app.command("export")
def document_export(
    output_path: Path = typer.Argument(..., help="Output file path"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    format: str = typer.Option("csv", "--format", "-f", help="Export format: csv, json")
):
    """Export documents to file"""
    
    monjyu = get_monjyu(config)
    documents = monjyu.list_documents(limit=10000)
    
    if format == "csv":
        import csv
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "id", "title", "authors", "year", "doi", "chunk_count"
            ])
            writer.writeheader()
            for doc in documents:
                writer.writerow({
                    "id": doc.id,
                    "title": doc.title,
                    "authors": "; ".join(doc.authors),
                    "year": doc.year,
                    "doi": doc.doi,
                    "chunk_count": doc.chunk_count
                })
    else:
        import json
        with open(output_path, "w") as f:
            json.dump([
                {
                    "id": d.id,
                    "title": d.title,
                    "authors": d.authors,
                    "year": d.year,
                    "doi": d.doi,
                    "chunk_count": d.chunk_count
                }
                for d in documents
            ], f, indent=2)
    
    console.print(f"[green]Exported {len(documents)} documents to {output_path}[/green]")
```

### 3.5 Citation コマンド

```python
# monjyu/cli/commands/citation.py

from pathlib import Path
from typing import Optional
import typer
from rich.table import Table
from rich.tree import Tree

from ..main import citation_app, console, get_monjyu, OutputFormat

@citation_app.command("chain")
def citation_chain(
    document_id: str = typer.Argument(..., help="Document ID"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    depth: int = typer.Option(2, "--depth", "-d", help="Chain depth"),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", "-o")
):
    """Show citation chain for a document"""
    
    monjyu = get_monjyu(config)
    chain = monjyu.get_citation_chain(document_id, depth=depth)
    
    if output == OutputFormat.json:
        import json
        console.print_json(json.dumps(chain))
    else:
        doc = monjyu.get_document(document_id)
        title = doc.title if doc else document_id
        
        tree = Tree(f"[bold]{title}[/bold]")
        
        # Cites
        cites_branch = tree.add("[cyan]Cites[/cyan]")
        for cited_id in chain.get("cites", [])[:10]:
            cited_doc = monjyu.get_document(cited_id)
            cites_branch.add(cited_doc.title if cited_doc else cited_id)
        
        # Cited by
        cited_by_branch = tree.add("[green]Cited by[/green]")
        for citing_id in chain.get("cited_by", [])[:10]:
            citing_doc = monjyu.get_document(citing_id)
            cited_by_branch.add(citing_doc.title if citing_doc else citing_id)
        
        console.print(tree)


@citation_app.command("related")
def citation_related(
    document_id: str = typer.Argument(..., help="Document ID"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    top_k: int = typer.Option(10, "--top-k", "-k"),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", "-o")
):
    """Find related papers"""
    
    monjyu = get_monjyu(config)
    related = monjyu.find_related_papers(document_id, top_k=top_k)
    
    if output == OutputFormat.json:
        import json
        console.print_json(json.dumps([
            {
                "document_id": r.document_id,
                "title": r.title,
                "relationship": r.relationship_type,
                "score": r.similarity_score
            }
            for r in related
        ]))
    else:
        table = Table(title="Related Papers")
        table.add_column("Title", max_width=50)
        table.add_column("Relationship", justify="center")
        table.add_column("Score", justify="right")
        
        for paper in related:
            table.add_row(
                paper.title[:50] + ("..." if len(paper.title) > 50 else ""),
                paper.relationship_type,
                f"{paper.similarity_score:.3f}"
            )
        
        console.print(table)


@citation_app.command("top")
def citation_top(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    top_k: int = typer.Option(10, "--top-k", "-k"),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", "-o")
):
    """Show most influential papers"""
    
    monjyu = get_monjyu(config)
    citation_manager = monjyu.get_citation_network()
    top_papers = citation_manager.get_most_influential(top_k)
    
    if output == OutputFormat.json:
        import json
        console.print_json(json.dumps([
            {
                "document_id": m.document_id,
                "citations": m.citation_count,
                "pagerank": m.pagerank,
                "influence_score": m.influence_score
            }
            for m in top_papers
        ]))
    else:
        table = Table(title="Most Influential Papers")
        table.add_column("Rank", justify="right")
        table.add_column("Title", max_width=40)
        table.add_column("Citations", justify="right")
        table.add_column("PageRank", justify="right")
        table.add_column("Influence", justify="right")
        
        for i, metrics in enumerate(top_papers, 1):
            doc = monjyu.get_document(metrics.document_id)
            title = doc.title if doc else metrics.document_id
            
            table.add_row(
                str(i),
                title[:40] + ("..." if len(title) > 40 else ""),
                str(metrics.citation_count),
                f"{metrics.pagerank:.4f}",
                f"{metrics.influence_score:.4f}"
            )
        
        console.print(table)
```

### 3.6 Config コマンド

```python
# monjyu/cli/commands/config.py

from pathlib import Path
from typing import Optional
import typer
from rich.syntax import Syntax
from rich.panel import Panel

from ..main import config_app, console

DEFAULT_CONFIG = """# MONJYU Configuration

# Basic settings
output_path: ./output
environment: local  # local | azure

# Index levels to build
index_levels: [0, 1]

# Search settings
default_search_mode: lazy  # vector | lazy | auto
default_top_k: 10

# Document processing
chunk_size: 1200
chunk_overlap: 100

# Local environment (Ollama)
llm_model: llama3:8b-instruct-q4_K_M
embedding_model: nomic-embed-text
ollama_base_url: http://192.168.224.1:11434

# Azure environment (optional)
# azure_openai_endpoint: https://xxx.openai.azure.com/
# azure_openai_api_key: ${AZURE_OPENAI_API_KEY}
# azure_search_endpoint: https://xxx.search.windows.net/
# azure_search_api_key: ${AZURE_SEARCH_API_KEY}
"""

@config_app.command("init")
def config_init(
    output_path: Path = typer.Option(Path("./monjyu.yaml"), "--output", "-o"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing file")
):
    """Initialize configuration file"""
    
    if output_path.exists() and not force:
        console.print(f"[yellow]Config file already exists: {output_path}[/yellow]")
        console.print("Use --force to overwrite")
        raise typer.Exit(1)
    
    with open(output_path, "w") as f:
        f.write(DEFAULT_CONFIG)
    
    console.print(f"[green]Configuration file created: {output_path}[/green]")


@config_app.command("show")
def config_show(
    config: Optional[Path] = typer.Option(None, "--config", "-c")
):
    """Show current configuration"""
    
    config_path = config or Path("./monjyu.yaml")
    
    if not config_path.exists():
        console.print(f"[yellow]Config file not found: {config_path}[/yellow]")
        console.print("Run 'monjyu config init' to create one")
        raise typer.Exit(1)
    
    with open(config_path) as f:
        content = f.read()
    
    syntax = Syntax(content, "yaml", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title=str(config_path)))


@config_app.command("validate")
def config_validate(
    config: Optional[Path] = typer.Option(None, "--config", "-c")
):
    """Validate configuration file"""
    
    config_path = config or Path("./monjyu.yaml")
    
    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        raise typer.Exit(1)
    
    try:
        from monjyu import MONJYU
        monjyu = MONJYU(config_path)
        
        console.print("[green]✓ Configuration is valid[/green]")
        console.print(f"  Environment: {monjyu.config.environment}")
        console.print(f"  Output path: {monjyu.config.output_path}")
        console.print(f"  Index levels: {monjyu.config.index_levels}")
    
    except Exception as e:
        console.print(f"[red]✗ Configuration error: {e}[/red]")
        raise typer.Exit(1)
```

### 3.7 エントリーポイント

```python
# monjyu/cli/__init__.py

from .main import app

def main():
    """CLI entry point"""
    app()

if __name__ == "__main__":
    main()
```

```toml
# pyproject.toml (抜粋)

[project.scripts]
monjyu = "monjyu.cli:main"
```

---

## 4. 設定

CLIの設定はPython APIと共通の `monjyu.yaml` を使用。

---

## 5. 使用例

```bash
# === インデックス ===

# ドキュメントをインデックス化
monjyu index build ./papers/

# Level 0 のみ
monjyu index build ./papers/ --levels 0

# 状態確認
monjyu index status

# JSON出力
monjyu index status --output json

# === 検索 ===

# 基本検索
monjyu search "What is Transformer?"

# ベクトル検索
monjyu search "BERT model" --mode vector

# JSON出力
monjyu search "GPT architecture" --output json

# 対話モード
monjyu search interactive

# === ドキュメント ===

# 一覧表示
monjyu document list
monjyu document list --limit 50

# 詳細表示
monjyu document show doc_001

# エクスポート
monjyu document export ./documents.csv

# === 引用ネットワーク ===

# 引用チェーン
monjyu citation chain doc_001 --depth 2

# 関連論文
monjyu citation related doc_001 --top-k 5

# 影響力ランキング
monjyu citation top --top-k 10

# === 設定 ===

# 設定ファイル生成
monjyu config init

# 設定表示
monjyu config show

# 設定検証
monjyu config validate

# === ヘルプ ===

monjyu --help
monjyu search --help
monjyu index build --help
```

---

## 6. テスト計画

### 6.1 単体テスト

| テストケース | 対象 | 期待結果 |
|-------------|------|---------|
| test_index_build_command | index build | 正常終了 |
| test_search_command | search | 結果表示 |
| test_document_list | document list | テーブル表示 |
| test_config_init | config init | ファイル生成 |
| test_json_output | --output json | JSON出力 |

### 6.2 統合テスト

| テストケース | 対象 | 期待結果 |
|-------------|------|---------|
| test_cli_workflow | index → search | 全フロー成功 |
| test_interactive_mode | search interactive | 対話動作 |
| test_error_handling | エラーケース | 適切なエラー表示 |

---

## 7. 実装タスク

| タスクID | タスク | 見積もり | 依存 |
|----------|--------|---------|------|
| TASK-008-01 | メインアプリ構造 | 1h | - |
| TASK-008-02 | index コマンド | 2h | FEAT-007 |
| TASK-008-03 | search コマンド | 2h | FEAT-007 |
| TASK-008-04 | document コマンド | 2h | FEAT-007 |
| TASK-008-05 | citation コマンド | 2h | FEAT-007 |
| TASK-008-06 | config コマンド | 1h | - |
| TASK-008-07 | エラーハンドリング | 1h | TASK-008-01~06 |
| TASK-008-08 | テスト作成 | 2h | TASK-008-01~07 |
| TASK-008-09 | ドキュメント作成 | 1h | TASK-008-01~07 |
| **合計** | | **14h** | |

---

## 8. 受入基準

- [ ] `monjyu index build ./papers/` でインデックス構築できる
- [ ] `monjyu search "query"` で検索できる
- [ ] `monjyu search interactive` で対話モードが動作する
- [ ] `monjyu document list` でドキュメント一覧を表示できる
- [ ] `monjyu citation chain doc_id` で引用チェーンを表示できる
- [ ] `--output json` でJSON出力できる
- [ ] `monjyu config init` で設定ファイルを生成できる
- [ ] エラー時に適切なメッセージを表示する
- [ ] `--help` で各コマンドのヘルプを表示できる
