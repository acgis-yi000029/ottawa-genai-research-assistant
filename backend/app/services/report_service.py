"""
Report Service

Handles report generation and management through the repository layer.
Demonstrates the complete Service → Repository → monk/ architecture.
"""

import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.models.report import Report, ReportMetadata
from app.repositories.report_repository import ReportRepository


class ReportService:
    """Service for handling report operations through repository layer."""

    def __init__(self, settings: Settings | None = None):
        self.report_repo = ReportRepository()
        self.settings = settings or get_settings()
        self.export_dir = Path("monk/reports/exports")
        self.export_dir.mkdir(parents=True, exist_ok=True)

    async def generate_report(
        self,
        query: str,
        language: str | None = "en",
        include_charts: bool = True,
        document_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate a comprehensive report based on a query."""
        try:
            # This is a simplified implementation
            # In a real system, this would analyze documents and generate content
            
            report_title = f"Report: {query}"
            
            # Generate mock content based on query
            content = await self._generate_report_content(query, language, document_ids)
            
            # Create report sections
            sections_data = [
                {
                    "title": "Executive Summary",
                    "content": content.get("summary", ""),
                    "charts": content.get("summary_charts", []) if include_charts else [],
                },
                {
                    "title": "Detailed Analysis", 
                    "content": content.get("analysis", ""),
                    "charts": content.get("analysis_charts", []) if include_charts else [],
                },
                {
                    "title": "Key Findings",
                    "content": content.get("findings", ""),
                    "charts": content.get("findings_charts", []) if include_charts else [],
                },
                {
                    "title": "Recommendations",
                    "content": content.get("recommendations", ""),
                    "charts": content.get("recommendation_charts", []) if include_charts else [],
                },
            ]
            
            return {
                "title": report_title,
                "brief_summary": content.get("brief_summary", ""),
                "summary": content.get("summary", ""),
                "analysis": content.get("analysis", ""),
                "findings": content.get("findings", ""),
                "recommendations": content.get("recommendations", ""),
                "summary_charts": content.get("summary_charts", []),
                "analysis_charts": content.get("analysis_charts", []),
                "findings_charts": content.get("findings_charts", []),
                "recommendation_charts": content.get("recommendation_charts", []),
                "sources": content.get("sources", []),
                "sections": sections_data,
            }
            
        except Exception as e:
            raise Exception(f"Error generating report: {str(e)}")

    async def _generate_report_content(
        self, query: str, language: str, document_ids: list[str] | None = None
    ) -> dict[str, Any]:
        """Generate report content based on query and available documents."""
        
        # Mock content generation - in reality this would analyze documents
        if language == "fr":
            content = {
                "brief_summary": f"Résumé exécutif pour: {query}",
                "summary": f"Ce rapport analyse {query} basé sur les données disponibles...",
                "analysis": f"L'analyse détaillée de {query} révèle plusieurs tendances importantes...",
                "findings": f"Les principales conclusions concernant {query} incluent...",
                "recommendations": f"Basé sur l'analyse de {query}, nous recommandons...",
                "sources": ["Données locales", "Rapports municipaux", "Statistiques officielles"],
            }
        else:
            content = {
                "brief_summary": f"Executive summary for: {query}",
                "summary": f"This report analyzes {query} based on available data...",
                "analysis": f"Detailed analysis of {query} reveals several important trends...",
                "findings": f"Key findings regarding {query} include...",
                "recommendations": f"Based on the analysis of {query}, we recommend...",
                "sources": ["Local Data", "Municipal Reports", "Official Statistics"],
            }
        
        # Add mock charts
        content.update({
            "summary_charts": [
                {
                    "type": "bar",
                    "title": "Overview Metrics",
                    "data": {"labels": ["Q1", "Q2", "Q3"], "values": [100, 120, 135]}
                }
            ],
            "analysis_charts": [
                {
                    "type": "line", 
                    "title": "Trend Analysis",
                    "data": {"labels": ["Jan", "Feb", "Mar"], "values": [85, 92, 78]}
                }
            ],
            "findings_charts": [],
            "recommendation_charts": [],
        })
        
        return content

    async def export_report(self, report: Report, format: str) -> dict[str, Any]:
        """Export a report to the specified format."""
        try:
            # Create export filename
            safe_title = "".join(c for c in report.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_title}_{timestamp}.{self._get_file_extension(format)}"
            file_path = self.export_dir / filename
            
            # Generate content based on format
            if format == "html":
                content = await self._generate_html_export(report)
            elif format == "pdf":
                content = await self._generate_pdf_export(report)
            elif format == "word":
                content = await self._generate_word_export(report)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            # Save file
            if format == "html":
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
            else:
                with open(file_path, "wb") as f:
                    f.write(content)
            
            # Return export info
            expires_at = datetime.now(timezone.utc) + timedelta(days=7)  # Files expire after 7 days
            
            return {
                "file_path": str(file_path),
                "filename": filename,
                "format": format,
                "size": os.path.getsize(file_path),
                "expires_at": expires_at.isoformat(),
            }
            
        except Exception as e:
            raise Exception(f"Error exporting report: {str(e)}")

    async def get_export_file_path(self, report_id: str, format: str) -> str | None:
        """Get the file path for an exported report."""
        try:
            # Look for existing export files
            pattern = f"*_{report_id[:8]}*.{self._get_file_extension(format)}"
            matching_files = list(self.export_dir.glob(pattern))
            
            if matching_files:
                # Return the most recent file
                latest_file = max(matching_files, key=os.path.getctime)
                return str(latest_file)
            
            return None
            
        except Exception:
            return None

    def _get_file_extension(self, format: str) -> str:
        """Get file extension for the given format."""
        extensions = {
            "pdf": "pdf",
            "word": "docx",
            "html": "html",
        }
        return extensions.get(format, "txt")

    async def _generate_html_export(self, report: Report) -> str:
        """Generate HTML export of the report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                .metadata {{ background: #f8f9fa; padding: 15px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>{report.title}</h1>
            <div class="metadata">
                <p><strong>Created:</strong> {report.created_at.strftime('%Y-%m-%d %H:%M')}</p>
                <p><strong>Status:</strong> {report.status}</p>
                <p><strong>Language:</strong> {report.language}</p>
            </div>
            <h2>Summary</h2>
            <p>{report.summary}</p>
            <h2>Content</h2>
            <div>{report.content}</div>
        </body>
        </html>
        """
        return html_content

    async def _generate_pdf_export(self, report: Report) -> bytes:
        """Generate PDF export of the report."""
        # For now, return HTML content as bytes
        # In production, you would use a library like reportlab or weasyprint
        html_content = await self._generate_html_export(report)
        return html_content.encode('utf-8')

    async def _generate_word_export(self, report: Report) -> bytes:
        """Generate Word document export of the report."""
        # For now, return HTML content as bytes
        # In production, you would use python-docx
        html_content = await self._generate_html_export(report)
        return html_content.encode('utf-8')

    async def create_report(
        self,
        title: str,
        description: str,
        report_type: str,
        author: str,
        content: str = "",
        status: str = "draft",
    ) -> Report:
        """Create a new report."""

        # Create report metadata
        metadata = ReportMetadata(
            word_count=len(content.split()) if content else 0,
            section_count=content.count("\n\n") + 1 if content else 0,
        )

        # Create new report
        report = Report(
            id=str(uuid.uuid4()),
            title=title,
            description=description,
            type=report_type,
            status=status,
            author=author,
            content=content,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            metadata=metadata,
            tags=[],
            version=1,
        )

        # Save through repository
        saved_report = self.report_repo.create(report)
        return saved_report

    async def get_report_by_id(self, report_id: str) -> Report | None:
        """Get report by ID."""
        return self.report_repo.get_by_id(report_id)

    async def list_reports(self) -> list[Report]:
        """Get all reports."""
        return self.report_repo.get_all()

    async def get_reports_by_type(self, report_type: str) -> list[Report]:
        """Get reports by type."""
        return self.report_repo.find_by_type(report_type)

    async def get_reports_by_status(self, status: str) -> list[Report]:
        """Get reports by status."""
        return self.report_repo.find_by_status(status)

    async def get_reports_by_author(self, author: str) -> list[Report]:
        """Get reports by author."""
        return self.report_repo.find_by_author(author)

    async def update_report(self, report_id: str, **kwargs) -> Report | None:
        """Update report information."""
        report = self.report_repo.get_by_id(report_id)
        if not report:
            return None

        # Update fields
        for field, value in kwargs.items():
            if hasattr(report, field):
                setattr(report, field, value)

        # Update metadata if content changed
        if "content" in kwargs:
            content = kwargs["content"]
            report.metadata.word_count = len(content.split()) if content else 0
            report.metadata.section_count = content.count("\n\n") + 1 if content else 0

        # Update timestamps and version
        report.updated_at = datetime.now(timezone.utc)
        report.version += 1

        # Save through repository
        return self.report_repo.update(report_id, report)

    async def update_report_content(
        self, report_id: str, content: str
    ) -> Report | None:
        """Update report content specifically."""
        return await self.update_report(report_id, content=content)

    async def update_report_status(self, report_id: str, status: str) -> Report | None:
        """Update report status."""
        return await self.update_report(report_id, status=status)

    async def publish_report(self, report_id: str) -> Report | None:
        """Publish a report (change status to published)."""
        return await self.update_report_status(report_id, "published")

    async def archive_report(self, report_id: str) -> Report | None:
        """Archive a report (change status to archived)."""
        return await self.update_report_status(report_id, "archived")

    async def delete_report(self, report_id: str) -> bool:
        """Delete a report."""
        return self.report_repo.delete(report_id)

    async def get_reports_summary(self) -> dict[str, Any]:
        """Get summary statistics of all reports."""
        try:
            all_reports = self.report_repo.get_all()

            # Calculate statistics
            total_reports = len(all_reports)
            status_counts = {}
            type_counts = {}
            author_counts = {}

            for report in all_reports:
                # Count by status
                status_counts[report.status] = status_counts.get(report.status, 0) + 1

                # Count by type
                type_counts[report.type] = type_counts.get(report.type, 0) + 1

                # Count by author
                author_counts[report.author] = author_counts.get(report.author, 0) + 1

            return {
                "total_reports": total_reports,
                "status_breakdown": status_counts,
                "type_breakdown": type_counts,
                "author_breakdown": author_counts,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            raise Exception(f"Error generating reports summary: {str(e)}")

    async def search_reports(self, query: str) -> list[Report]:
        """Search reports by title, description, or content."""
        try:
            all_reports = self.report_repo.get_all()
            query_lower = query.lower()

            matching_reports = []
            for report in all_reports:
                # Search in title, description, and content
                if (
                    query_lower in report.title.lower()
                    or query_lower in report.description.lower()
                    or query_lower in report.content.lower()
                ):
                    matching_reports.append(report)

            return matching_reports

        except Exception as e:
            raise Exception(f"Error searching reports: {str(e)}")

    async def get_recent_reports(self, limit: int = 10) -> list[Report]:
        """Get most recently updated reports."""
        try:
            all_reports = self.report_repo.get_all()

            # Sort by updated_at in descending order
            sorted_reports = sorted(
                all_reports, key=lambda r: r.updated_at, reverse=True
            )

            return sorted_reports[:limit]

        except Exception as e:
            raise Exception(f"Error retrieving recent reports: {str(e)}")

    async def duplicate_report(self, report_id: str, new_title: str) -> Report | None:
        """Create a copy of an existing report."""
        try:
            original_report = self.report_repo.get_by_id(report_id)
            if not original_report:
                return None

            # Create duplicate with new ID and title
            duplicate = await self.create_report(
                title=new_title,
                description=f"Copy of: {original_report.description}",
                report_type=original_report.type,
                author=original_report.author,
                content=original_report.content,
                status="draft",
            )

            return duplicate

        except Exception as e:
            raise Exception(f"Error duplicating report: {str(e)}")
