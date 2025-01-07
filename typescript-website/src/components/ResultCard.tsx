import React from 'react';
import styles from './ResultCard.module.css';

interface ResultCardProps {
  result: any;
  type: 'image' | 'video' | 'factchecker';
}

const ResultCard: React.FC<ResultCardProps> = ({ result, type }) => {
  if (!result) {
    return null;
  }

  return (
    <div className={styles.resultCard}>
      <h2>Detection Result</h2>
      <p><strong>Type:</strong> {type === 'factchecker' ? 'Factchecker' : type === 'image' ? 'Image' : 'Video'}</p>
      {result.confidence && <p><strong>Confidence:</strong> {result.confidence.toFixed(2) * 100}%</p>}
      {result.prediction && <p><strong>Prediction:</strong> {result.prediction}</p>}
      {result.explanation && <p><strong>Explanation:</strong> {result.explanation}</p>}

      {type === 'factchecker' && result.articles && (
        <div className={styles.articles}>
          <h3>Source Articles</h3>
          <ul>
            {result.articles.map((article: string, index: number) => (
              <li key={index}>
                <a href={article} target="_blank" rel="noopener noreferrer">
                  {article}
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default ResultCard;
