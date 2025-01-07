import React, { useState } from 'react';
import DragAndDropUpload from '../components/DragAndDropUpload';
import ResultCard from '../components/ResultCard';
import Factchecker from '../components/Factchecker';
import styles from '../styles/Home.module.css';

const Home = () => {
  const [activeTab, setActiveTab] = useState<'image' | 'video' | 'factchecker'>('image');
  const [result, setResult] = useState<any>(null);

  const handleResult = (data: any) => {
    setResult(data);
  };

  return (
    <div className={styles.container}>
      {/* Left Section: Image */}
      <div className={styles.imageSection}>
        <img src="/image.png" alt="App Logo" className={styles.logoImage} />
      </div>

      {/* Right Section: Tabs and Content */}
      <div className={styles.contentSection}>
        <h1>GO AI Deepfake Detector & Fact-checker</h1>
        <div className={styles.tabs}>
          <button
            className={activeTab === 'image' ? styles.activeTab : ''}
            onClick={() => setActiveTab('image')}
          >
            Image Detection
          </button>
          <button
            className={activeTab === 'video' ? styles.activeTab : ''}
            onClick={() => setActiveTab('video')}
          >
            Video Detection
          </button>
          <button
            className={activeTab === 'factchecker' ? styles.activeTab : ''}
            onClick={() => setActiveTab('factchecker')}
          >
            Factchecker
          </button>
        </div>

        {activeTab === 'image' && (
          <DragAndDropUpload
            endpoint="https://fastapi-whatsapp-17290362188.us-central1.run.app/predict-image"
            onResult={handleResult}
          />
        )}
        {activeTab === 'video' && (
          <DragAndDropUpload
            endpoint="https://fastapi-whatsapp-17290362188.us-central1.run.app/predict-video"
            onResult={handleResult}
          />
        )}
        {activeTab === 'factchecker' && <Factchecker onResult={handleResult} />}
        {result && <ResultCard result={result} type={activeTab} />}
      </div>
    </div>
  );
};

export default Home;
