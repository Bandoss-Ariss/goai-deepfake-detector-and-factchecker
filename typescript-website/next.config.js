module.exports = {
    async rewrites() {
      return [
        {
          source: '/api/predict-image',
          destination: 'http://fastapi-whatsapp-17290362188.us-central1.run.app/predict-image',
        },
        {
          source: '/api/predict-videos',
          destination: 'http://fastapi-whatsapp-17290362188.us-central1.run.app/predict-video',
        },
        {
          source: '/api/verify-claim',
          destination: 'http://u-factchecker-1009734859869.us-central1.run.app/verify-claim/',
        },
      ];
    },
  };
  