import { CheckCircle, FileText, Upload, X } from 'lucide-react';
import React, { useEffect, useRef, useState } from 'react';
import { UploadedFile } from '../mock/types';
import { realApi } from '../services/api';
import './DocumentUploadPage.css';

const DocumentUploadPage: React.FC = () => {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load existing files on component mount
  useEffect(() => {
    loadUploadedFiles();
  }, []);

  const loadUploadedFiles = async () => {
    try {
      const uploadedFiles = await realApi.getUploadedFiles();
      setFiles(uploadedFiles);
    } catch (error) {
      console.error('Error loading files:', error);
      // Show empty array if API fails
      setFiles([]);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const droppedFiles = Array.from(e.dataTransfer.files);
    processFiles(droppedFiles);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    processFiles(selectedFiles);
  };

  const processFiles = async (selectedFiles: File[]) => {
    // Filter for PDF files only
    const pdfFiles = selectedFiles.filter(file => file.type === 'application/pdf');
    
    if (pdfFiles.length !== selectedFiles.length) {
      alert('Only PDF files are supported. Some files were skipped.');
    }

    if (pdfFiles.length === 0) {
      return;
    }

    setIsLoading(true);

    // Process each file
    for (const file of pdfFiles) {
      const tempFile: UploadedFile = {
        id: Date.now().toString() + Math.random(),
        name: file.name,
        size: file.size,
        type: file.type,
        status: 'uploading',
        progress: 0,
        uploadedAt: new Date()
      };

      // Add to UI immediately
      setFiles(prev => [...prev, tempFile]);

      try {
        // Use real API to upload file
        const uploadedFile = await realApi.uploadFile(file);
        
        // Update the file status to completed
        setFiles(prev => prev.map(f => 
          f.id === tempFile.id 
            ? { ...uploadedFile, status: 'completed', progress: 100 }
            : f
        ));
      } catch (error) {
        console.error('Upload failed:', error);
        
        // Update the file status to error
        setFiles(prev => prev.map(f => 
          f.id === tempFile.id 
            ? { ...f, status: 'error', progress: 0 }
            : f
        ));
      }
    }

    setIsLoading(false);
  };

  const removeFile = async (fileId: string) => {
    try {
      await realApi.deleteFile(fileId);
      setFiles(prev => prev.filter(file => file.id !== fileId));
    } catch (error) {
      console.error('Error deleting file:', error);
      alert('Failed to delete document on server. Please try again.');
    }
  };

  const openFilePicker = () => {
    fileInputRef.current?.click();
  };

  const getStatusIcon = (status: UploadedFile['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="status-icon completed" size={20} />;
      case 'error':
        return <X className="status-icon error" size={20} />;
      default:
        return <div className="upload-spinner" />;
    }
  };

  return (
    <div className="upload-page">
      <div className="upload-container">
        <div className="upload-header">
          <h1>Document Upload</h1>
          <p>Upload PDF reports to expand the AI knowledge base</p>
        </div>

        {/* Upload Zone */}
        <div 
          className={`upload-zone ${isDragOver ? 'drag-over' : ''} ${isLoading ? 'uploading' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={isLoading ? undefined : openFilePicker}
          role="button"
          tabIndex={0}
          aria-label="Upload documents"
        >
          <input type="file" ref={fileInputRef} onChange={handleFileSelect} multiple accept=".pdf" />
          
          <div className="upload-content">
            <Upload size={48} className="upload-icon" />
            <h3>{isLoading ? 'Uploading files...' : 'Drop PDF files here or click to browse'}</h3>
            <p>Supported format: PDF documents only</p>
            <button className="upload-btn" disabled={isLoading}>
              {isLoading ? 'Uploading...' : 'Choose Files'}
            </button>
          </div>
        </div>

        {/* Upload Progress */}
        {files.some(file => file.status === 'uploading') && (
          <div className="upload-progress-section">
            <h3>Uploading Files</h3>
            {files
              .filter(file => file.status === 'uploading')
              .map(file => (
                <div key={file.id} className="progress-item">
                  <div className="progress-info">
                    <FileText size={16} />
                    <span className="file-name">{file.name}</span>
                    <span className="progress-text">{file.progress}%</span>
                  </div>
                  <div className="progress-bar">
                    <div 
                      className="progress-fill"
                      style={{ width: `${file.progress}%` }}
                    />
                  </div>
                </div>
              ))}
          </div>
        )}

        {/* File List */}
        <div className="file-list-section">
          <div className="section-header">
            <h3>Uploaded Documents ({files.filter(f => f.status === 'completed').length})</h3>
            <p>These documents are available for AI analysis</p>
          </div>

          {files.length === 0 ? (
            <div className="empty-state">
              <FileText size={48} className="empty-icon" />
              <h4>No documents uploaded yet</h4>
              <p>Upload PDF reports to get started with AI analysis</p>
            </div>
          ) : (
            <div className="file-list">
              {files.map(file => (
                <div key={file.id} className="file-item">
                  <div className="file-info">
                    <div className="file-icon">
                      <FileText size={24} />
                    </div>
                    <div className="file-details">
                      <h4 className="file-name">{file.name}</h4>
                      <div className="file-meta">
                        <span className="file-size">{formatFileSize(file.size)}</span>
                        <span className="file-date">
                          Uploaded {file.uploadedAt.toLocaleDateString()}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="file-actions">
                    {getStatusIcon(file.status)}
                    
                    {file.status === 'completed' && (
                      <>
                        <button 
                          className="action-btn preview-btn"
                          aria-label={`Preview ${file.name}`}
                        >
                          {/* Clock icon removed as per new_code */}
                        </button>
                      </>
                    )}
                    
                    <button 
                      onClick={() => removeFile(file.id)}
                      className="action-btn remove-btn"
                      aria-label={`Remove ${file.name}`}
                    >
                      <X size={16} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Processing Status */}
        <div className="processing-status">
          <div className="status-card">
            <h4>📄 Document Processing</h4>
            <p>PDF text extraction and indexing</p>
            <div className="status-indicator completed">Completed</div>
          </div>
          
          <div className="status-card">
            <h4>�� Content Analysis</h4>
            <p>AI semantic understanding and categorization</p>
            <div className="status-indicator completed">Completed</div>
          </div>
          
          <div className="status-card">
            <h4>💾 Knowledge Base</h4>
            <p>Integration with search and Q&A system</p>
            <div className="status-indicator completed">Ready</div>
          </div>
        </div>

        {/* Usage Guidelines */}
        <div className="guidelines-section">
          <h3>Upload Guidelines</h3>
          <div className="guidelines-grid">
            <div className="guideline-item">
              <h4>✅ Supported Formats</h4>
              <ul>
                <li>PDF documents only</li>
                <li>Text-based PDFs (not scanned images)</li>
                <li>Maximum file size: 50MB</li>
              </ul>
            </div>
            
            <div className="guideline-item">
              <h4>📋 Best Practices</h4>
              <ul>
                <li>Use descriptive file names</li>
                <li>Ensure documents are not password protected</li>
                <li>Upload the most recent versions</li>
              </ul>
            </div>
            
            <div className="guideline-item">
              <h4>🔒 Privacy & Security</h4>
              <ul>
                <li>Only upload public information</li>
                <li>No personal or confidential data</li>
                <li>Documents are processed locally</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DocumentUploadPage; 