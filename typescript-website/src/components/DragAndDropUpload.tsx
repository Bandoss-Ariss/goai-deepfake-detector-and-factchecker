import React, { useState } from 'react';
import axios from 'axios';
import styles from './DragAndDropUpload.module.css';

interface DragAndDropUploadProps {
  endpoint: string;
  onResult: (result: any) => void;
}

const DragAndDropUpload: React.FC<DragAndDropUploadProps> = ({ endpoint, onResult }) => {
  const [file, setFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileUpload = async () => {
    if (!file) {
      setError('Please select a file first!');
      return;
    }

    setUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(endpoint, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      onResult(response.data);
    } catch (err) {
      setError('An error occurred while uploading the file.');
    } finally {
      setUploading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);

      // Generate an image preview URL
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(selectedFile);
    }
  };

  return (
    <div className={styles.dragAndDrop}>
      <label htmlFor="fileInput" className={styles.dropZone}>
        {imagePreview ? (
          <div className={styles.imagePreview}>
            <img src={imagePreview} alt="Preview" className={styles.previewImage} />
          </div>
        ) : (
          <p>Drag and drop a file here, or click to select a file.</p>
        )}
        <input
          id="fileInput"
          type="file"
          onChange={handleFileChange}
          className={styles.fileInput}
        />
      </label>
      {file && (
        <div className={styles.fileInfo}>
          <p>Selected file: {file.name}</p>
          <button onClick={handleFileUpload} disabled={uploading} className={styles.uploadButton}>
            {uploading ? 'We are processing your media. This may take a while. Please Hold on:)...' : 'Submit'}
          </button>
        </div>
      )}
      {error && <p className={styles.error}>{error}</p>}
    </div>
  );
};

export default DragAndDropUpload;
