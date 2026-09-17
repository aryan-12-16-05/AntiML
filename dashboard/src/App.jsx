import { useState, useEffect, useRef, useCallback } from 'react'
import { Routes, Route, useLocation, useNavigate } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import AlertsPage from './pages/Alerts'
import CustomersPage from './pages/Customers'
import ModelsPage from './pages/Models'

const WS_URL = 'ws://localhost:8000/ws/alerts'
const API_URL = 'http://localhost:8000'

// ------------------------------------------------------------------ //
//  WebSocket hook for live alerts                                      //
// ------------------------------------------------------------------ //
export function useAlertStream() {
  const [connected, setConnected] = useState(false)
  const [alerts, setAlerts] = useState([])
  const [toasts, setToasts] = useState([])
  const wsRef = useRef(null)

  const addToast = useCallback((msg, type = 'alert') => {
    const id = Date.now()
    setToasts(prev => [...prev, { id, msg, type }])
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 3000)
  }, [])

  useEffect(() => {
    let reconnectTimer

    function connect() {
      try {
        const ws = new WebSocket(WS_URL)
        wsRef.current = ws

        ws.onopen = () => {
          setConnected(true)
        }

        ws.onmessage = (e) => {
          const data = JSON.parse(e.data)
          if (data.type === 'initial_alerts') {
            setAlerts(data.data || [])
          } else if (data.type === 'new_alert') {
            setAlerts(prev => [data.data, ...prev].slice(0, 500))
            const a = data.data
            const sev = a.severity || 'MEDIUM'
            if (sev === 'CRITICAL' || sev === 'HIGH') {
              addToast(`⚠ ${sev}: ${a.detected_pattern} pattern — ${a.from_account?.slice(0,8)}`, 'alert')
            }
          }
        }

        ws.onclose = () => {
          setConnected(false)
          reconnectTimer = setTimeout(connect, 3000)
        }

        ws.onerror = () => ws.close()
      } catch (e) {
        reconnectTimer = setTimeout(connect, 3000)
      }
    }

    connect()
    return () => {
      clearTimeout(reconnectTimer)
      wsRef.current?.close()
    }
  }, [addToast])

  return { connected, alerts, setAlerts, toasts }
}

// ------------------------------------------------------------------ //
//  Global Context                                                      //
// ------------------------------------------------------------------ //
import { createContext, useContext } from 'react'
export const AppContext = createContext({})
export const useApp = () => useContext(AppContext)

// ------------------------------------------------------------------ //
//  Sidebar Navigation                                                  //
// ------------------------------------------------------------------ //
function Sidebar({ connected, pendingCount }) {
  const navigate = useNavigate()
  const location = useLocation()
  const page = location.pathname

  const nav = (path) => navigate(path)

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="logo-icon">🛡</div>
        <div>
          <div className="logo-text">AML Shield</div>
          <div className="logo-sub">AI Detection System</div>
        </div>
      </div>

      <nav className="sidebar-nav">
        <div className="nav-label">Overview</div>
        <button className={`nav-item${page === '/' ? ' active' : ''}`} onClick={() => nav('/')}>
          <span className="nav-icon">📊</span> Dashboard
        </button>

        <div className="nav-label">Detection</div>
        <button className={`nav-item${page === '/alerts' ? ' active' : ''}`} onClick={() => nav('/alerts')}>
          <span className="nav-icon">🚨</span> Alerts
          {pendingCount > 0 && <span className="nav-badge">{pendingCount}</span>}
        </button>
        <button className={`nav-item${page === '/customers' ? ' active' : ''}`} onClick={() => nav('/customers')}>
          <span className="nav-icon">👤</span> Customers
        </button>

        <div className="nav-label">System</div>
        <button className={`nav-item${page === '/models' ? ' active' : ''}`} onClick={() => nav('/models')}>
          <span className="nav-icon">🧠</span> Models
        </button>
      </nav>

      <div className="sidebar-footer">
        <div className="connection-status">
          <div className={`connection-dot${connected ? '' : ' disconnected'}`} />
          {connected ? 'Live Stream' : 'Reconnecting...'}
        </div>
      </div>
    </aside>
  )
}

// ------------------------------------------------------------------ //
//  Toast Notifications                                                 //
// ------------------------------------------------------------------ //
function ToastContainer({ toasts }) {
  return (
    <div className="toast-container">
      {toasts.map(t => (
        <div key={t.id} className={`toast ${t.type === 'alert' ? 'alert-toast' : 'success-toast'}`}>
          <span>{t.msg}</span>
        </div>
      ))}
    </div>
  )
}

// ------------------------------------------------------------------ //
//  App Root                                                            //
// ------------------------------------------------------------------ //
export default function App() {
  const { connected, alerts, setAlerts, toasts } = useAlertStream()
  const [stats, setStats] = useState({ total: 0, pending: 0, approved: 0, rejected: 0, retroactive: 0, critical: 0 })

  // Fetch stats periodically
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const r = await fetch(`${API_URL}/api/alerts/stats`)
        if (r.ok) setStats(await r.json())
      } catch {}
    }
    fetchStats()
    const interval = setInterval(fetchStats, 5000)
    return () => clearInterval(interval)
  }, [])

  const reviewAlert = useCallback(async (alertId, action, note = '', accounts = [], ttl = 30) => {
    try {
      const r = await fetch(`${API_URL}/api/decisions/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          alert_id: alertId,
          action,
          note,
          whitelist_accounts: accounts,
          whitelist_ttl_days: ttl,
        }),
      })
      const data = await r.json()
      if (data.success) {
        setAlerts(prev => prev.map(a => a.id === alertId ? { ...a, status: action } : a))
      }
      return data
    } catch (e) {
      console.error(e)
      return { success: false }
    }
  }, [setAlerts])

  const ctx = { alerts, stats, connected, reviewAlert, API_URL }

  return (
    <AppContext.Provider value={ctx}>
      <div className="app-shell">
        <Sidebar connected={connected} pendingCount={stats.pending} />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/alerts" element={<AlertsPage />} />
            <Route path="/customers" element={<CustomersPage />} />
            <Route path="/models" element={<ModelsPage />} />
          </Routes>
        </main>
      </div>
      <ToastContainer toasts={toasts} />
    </AppContext.Provider>
  )
}
