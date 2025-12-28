# MONJYU CLI - Document Commands
"""
FEAT-008: CLI - document コマンド群
ドキュメント一覧・詳細・エクスポート
"""

from pathlib import Path
from typing import Optional
import json
import csv

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from monjyu.cli.main import OutputFormat, get_monjyu, print_error, print_success

console = Console()
document_app = typer.Typer(help="Document management commands")


@document_app.command("list")
def document_list(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
    limit: int = typer.Option(
        20, "--limit", "-n", help="Maximum number of documents"
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", "-o", help="Output format"
    ),
):
    """List documents in the index"""
    
    try:
        monjyu = get_monjyu(config)
        documents = monjyu.list_documents(limit=limit)
        
        if output == OutputFormat.json:
            console.print_json(json.dumps([
                {
                    "id": d.id,
                    "title": d.title,
                    "authors": d.authors,
                    "year": d.year,
                    "doi": d.doi,
                    "chunk_count": d.chunk_count,
                    "citation_count": d.citation_count,
                }
                for d in documents
            ]))
        else:
            if not documents:
                console.print("[yellow]No documents found in index[/yellow]")
                console.print("Run 'monjyu index build <path>' to index documents")
                return
            
            table = Table(title=f"Documents ({len(documents)} shown)")
            table.add_column("ID", style="dim", max_width=10)
            table.add_column("Title", max_width=45)
            table.add_column("Authors", max_width=25)
            table.add_column("Year", justify="center", width=6)
            table.add_column("Chunks", justify="right", width=7)
            
            for doc in documents:
                # 著者の省略表示
                authors = ", ".join(doc.authors[:2]) if doc.authors else "-"
                if len(doc.authors) > 2:
                    authors += f" +{len(doc.authors) - 2}"
                
                # タイトルの省略
                title = doc.title[:45]
                if len(doc.title) > 45:
                    title += "..."
                
                table.add_row(
                    doc.id[:10] + "..." if len(doc.id) > 10 else doc.id,
                    title,
                    authors[:25],
                    str(doc.year) if doc.year else "-",
                    str(doc.chunk_count),
                )
            
            console.print(table)
            
            status = monjyu.get_status()
            if status.document_count > limit:
                console.print(
                    f"\n[dim]Showing {limit} of {status.document_count} documents. "
                    f"Use --limit to show more.[/dim]"
                )
    
    except Exception as e:
        print_error(f"Failed to list documents: {e}")
        raise typer.Exit(1)


@document_app.command("show")
def document_show(
    document_id: str = typer.Argument(..., help="Document ID"),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", "-o", help="Output format"
    ),
):
    """Show document details"""
    
    try:
        monjyu = get_monjyu(config)
        doc = monjyu.get_document(document_id)
        
        if doc is None:
            print_error(f"Document not found: {document_id}")
            raise typer.Exit(1)
        
        if output == OutputFormat.json:
            console.print_json(json.dumps({
                "id": doc.id,
                "title": doc.title,
                "authors": doc.authors,
                "year": doc.year,
                "doi": doc.doi,
                "abstract": doc.abstract,
                "chunk_count": doc.chunk_count,
                "citation_count": doc.citation_count,
                "reference_count": doc.reference_count,
                "influence_score": doc.influence_score,
            }))
        else:
            # 基本情報
            authors_str = ", ".join(doc.authors) if doc.authors else "Unknown"
            
            content = (
                f"[bold]{doc.title}[/bold]\n\n"
                f"[cyan]ID:[/cyan] {doc.id}\n"
                f"[cyan]Authors:[/cyan] {authors_str}\n"
                f"[cyan]Year:[/cyan] {doc.year or 'Unknown'}\n"
                f"[cyan]DOI:[/cyan] {doc.doi or 'None'}\n"
            )
            
            # アブストラクト
            if doc.abstract:
                abstract = doc.abstract[:300]
                if len(doc.abstract) > 300:
                    abstract += "..."
                content += f"\n[cyan]Abstract:[/cyan]\n{abstract}\n"
            
            # 統計情報
            content += (
                f"\n[bold]Index Statistics:[/bold]\n"
                f"  Chunks: {doc.chunk_count}\n"
            )
            
            # 引用情報
            content += (
                f"\n[bold]Citation Metrics:[/bold]\n"
                f"  Cited by: {doc.citation_count}\n"
                f"  References: {doc.reference_count}\n"
                f"  Influence Score: {doc.influence_score:.4f}"
            )
            
            console.print(Panel(content, title="Document Details"))
    
    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Failed to get document: {e}")
        raise typer.Exit(1)


@document_app.command("export")
def document_export(
    output_path: Path = typer.Argument(..., help="Output file path"),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
    format: str = typer.Option(
        "csv", "--format", "-f", help="Export format: csv, json"
    ),
    limit: int = typer.Option(
        10000, "--limit", "-n", help="Maximum documents to export"
    ),
):
    """Export documents to file"""
    
    try:
        monjyu = get_monjyu(config)
        documents = monjyu.list_documents(limit=limit)
        
        if not documents:
            console.print("[yellow]No documents to export[/yellow]")
            raise typer.Exit(0)
        
        if format.lower() == "csv":
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "id", "title", "authors", "year", "doi", 
                    "chunk_count", "citation_count"
                ])
                writer.writeheader()
                for doc in documents:
                    writer.writerow({
                        "id": doc.id,
                        "title": doc.title,
                        "authors": "; ".join(doc.authors) if doc.authors else "",
                        "year": doc.year or "",
                        "doi": doc.doi or "",
                        "chunk_count": doc.chunk_count,
                        "citation_count": doc.citation_count,
                    })
        
        elif format.lower() == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump([
                    {
                        "id": d.id,
                        "title": d.title,
                        "authors": d.authors,
                        "year": d.year,
                        "doi": d.doi,
                        "chunk_count": d.chunk_count,
                        "citation_count": d.citation_count,
                    }
                    for d in documents
                ], f, indent=2, ensure_ascii=False)
        
        else:
            print_error(f"Unsupported format: {format}")
            console.print("Supported formats: csv, json")
            raise typer.Exit(1)
        
        print_success(f"Exported {len(documents)} documents to {output_path}")
    
    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Export failed: {e}")
        raise typer.Exit(1)


@document_app.command("search")
def document_search(
    pattern: str = typer.Argument(..., help="Search pattern (title, author, etc.)"),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
    limit: int = typer.Option(
        20, "--limit", "-n", help="Maximum results"
    ),
):
    """Search documents by title or author"""
    
    try:
        monjyu = get_monjyu(config)
        documents = monjyu.list_documents(limit=1000)
        
        # 簡易検索（タイトルと著者名で部分一致）
        pattern_lower = pattern.lower()
        matches = []
        
        for doc in documents:
            if pattern_lower in doc.title.lower():
                matches.append(doc)
            elif any(pattern_lower in a.lower() for a in doc.authors):
                matches.append(doc)
        
        matches = matches[:limit]
        
        if not matches:
            console.print(f"[yellow]No documents matching '{pattern}'[/yellow]")
            return
        
        table = Table(title=f"Search Results for '{pattern}'")
        table.add_column("ID", style="dim", max_width=10)
        table.add_column("Title", max_width=50)
        table.add_column("Year", justify="center")
        
        for doc in matches:
            table.add_row(
                doc.id[:10],
                doc.title[:50] + ("..." if len(doc.title) > 50 else ""),
                str(doc.year) if doc.year else "-",
            )
        
        console.print(table)
        console.print(f"\n[dim]Found {len(matches)} matching documents[/dim]")
    
    except Exception as e:
        print_error(f"Search failed: {e}")
        raise typer.Exit(1)
