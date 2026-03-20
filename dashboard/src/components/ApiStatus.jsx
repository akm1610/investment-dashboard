import { useState, useEffect } from 'react';
import { fetchHealth } from '../services/api';

export default function ApiStatus() {
  const [status, setStatus] = useState('checking');

  useEffect(() => {
    let cancelled = false;

    function check() {
      fetchHealth()
        .then(() => { if (!cancelled) setStatus('online'); })
        .catch(() => { if (!cancelled) setStatus('offline'); });
    }

    check();
    const interval = setInterval(check, 30_000);
    return () => { cancelled = true; clearInterval(interval); };
  }, []);

  const dot = status === 'online' ? '🟢' : status === 'offline' ? '🔴' : '🟡';

  return (
    <div className="api-status">
      <span>{dot}</span>
      <span>API {status === 'checking' ? 'connecting…' : status}</span>
    </div>
  );
}
