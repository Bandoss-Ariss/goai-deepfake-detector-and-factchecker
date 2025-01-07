import React, { useState } from 'react';
import axios from 'axios';
import styles from './Factchecker.module.css';

const Factchecker: React.FC<{ onResult: (result: any) => void }> = ({ onResult }) => {
  const [inputText, setInputText] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Handle file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setInputText(''); // Clear text input if a file is selected
    }
  };

  // Handle text input
  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputText(e.target.value);
    setFile(null); // Clear file input if text is entered
  };

  // Handle the submission
  const handleVerifyClaim = async () => {
    if (!inputText && !file) {
      setError('Please provide a text claim or upload an image.');
      return;
    }
  
    setUploading(true);
    setError(null);
  
    try {
      const formData = new FormData();
      if (inputText) {
        formData.append('text', inputText);
      }
      if (file) {
        formData.append('image', file); // Make sure this key matches your API
      }
  
      const response = await axios.post(
        'https://u-factchecker-1009734859869.us-central1.run.app/verify-claim/',
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
        }
      );
  
      onResult(response.data);
    } catch (err) {
      setError('An error occurred while verifying the claim.');
    } finally {
      setUploading(false);
    }
  };  

  return (
    <div className={styles.factchecker}>
      <h2>Factchecker</h2>

      {/* Text Input */}
      <textarea
        value={inputText}
        onChange={handleTextChange}
        placeholder="Enter a claim to verify..."
        className={styles.textArea}
        disabled={!!file}
      ></textarea>

      <p>OR</p>

      {/* File Input with Custom Button */}
      <label htmlFor="fileInput" className={styles.uploadLabel}>
        {file ? `Selected file: ${file.name}` : 'Click to upload an image'}
      </label>
      <input
        id="fileInput"
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        className={styles.fileInput}
      />

      <button onClick={handleVerifyClaim} disabled={uploading} className={styles.uploadButton}>
        {uploading ? 'Verifying. Please hold on :)' : 'Verify Claim'}
      </button>

      {error && <p className={styles.error}>{error}</p>}
    </div>
  );
};

export default Factchecker;
