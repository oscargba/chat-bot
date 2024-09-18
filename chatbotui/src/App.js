import React, { useEffect } from 'react';
import Chatbot from './components/Chatbot';

function App() {
  console.log(`[APP***]`);
  useEffect(() => {
    // Place any code here that you want to run when the App component mounts
  }, []);
  return (
    <div>
      <Chatbot/>
    </div>
  );
}

export default App;
