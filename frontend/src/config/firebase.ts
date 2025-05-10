import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getAnalytics } from 'firebase/analytics';

const firebaseConfig = {
  apiKey: "AIzaSyBnkKLHhPtAvMbM5Kd3PBzpDXrjhQyNdPw",
  authDomain: "time-series-forecasting-d519a.firebaseapp.com",
  projectId: "time-series-forecasting-d519a",
  storageBucket: "time-series-forecasting-d519a.firebasestorage.app",
  messagingSenderId: "82213961000",
  appId: "1:82213961000:web:65b5310880015cd4aca41d",
  measurementId: "G-21MKRRY2FP"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Firebase services
export const auth = getAuth(app);
export const analytics = getAnalytics(app);

export default app;
