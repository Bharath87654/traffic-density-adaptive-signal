import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import './index.css'; // <-- THIS IS CRUCIAL
import React from 'react'
import ReactDOM from 'react-dom/client'
// ... rest of the file

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
