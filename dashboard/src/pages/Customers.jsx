import { useState, useEffect } from 'react'
import { useApp } from '../App'

export default function CustomersPage() {
  const { API_URL } = useApp()
  const [search, setSearch] = useState('')
  const [profiles, setProfiles] = useState([])
  const [whitelist, setWhitelist] = useState([])
  const [loading, setLoading] = useState(false)
  const [selected, setSelected] = useState(null)

  const fetchProfiles = async (q = '') => {
    setLoading(true)
    try {
      const r = await fetch(`${API_URL}/api/profiles/?query=${encodeURIComponent(q)}&limit=50`)
      if (r.ok) {
        const data = await r.json()
        setProfiles(data.profiles || [])
      }
    } catch {}
    setLoading(false)
  }

  const fetchWhitelist = async () => {
    try {
      const r = await fetch(`${API_URL}/api/decisions/whitelist`)
      if (r.ok) {
        const data = await r.json()
        setWhitelist(data.entries || [])
      }
    } catch {}
  }

  useEffect(() => { fetchProfiles(); fetchWhitelist() }, [])

  const handleSearch = (e) => {
    setSearch(e.target.value)
    const val = e.target.value
    const timeout = setTimeout(() => fetchProfiles(val), 400)
    return () => clearTimeout(timeout)
  }

  const revokeWhitelist = async (account) => {
    try {
      await fetch(`${API_URL}/api/decisions/whitelist/${account}`, { method: 'DELETE' })
      fetchWhitelist()
    } catch {}
  }

  const RISK_COLOR = { HIGH: '#ef4444', MEDIUM: '#f59e0b', LOW: '#10b981' }
  const ENTITY_ICON = {
    Corporation: '🏢', Partnership: '🤝', 'Sole Proprietorship': '👤',
    Individual: '👥', Country: '🌍', Direct: '🔗', Unknown: '❓'
  }

  return (
    <div>
      <div className="page-header">
        <div>
          <div className="page-title">Customer Profiles</div>
          <div className="page-subtitle">Behavioral baselines &amp; whitelist management</div>
        </div>
      </div>

      <div className="page-body">

        {/* Whitelist Panel */}
        {whitelist.length > 0 && (
          <div className="card">
            <div className="card-header">
              <span className="card-title">✅ Active Whitelist ({whitelist.length})</span>
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Accounts suppressed from detection</span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {whitelist.map(entry => (
                <div key={entry.id} style={{
                  display: 'flex', alignItems: 'center', gap: 12,
                  padding: '10px 14px', background: 'var(--bg-secondary)',
                  borderRadius: 'var(--radius-sm)', border: '1px solid rgba(16,185,129,0.2)'
                }}>
                  <div style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--green)', boxShadow: '0 0 6px var(--green)' }} />
                  <span className="account-chip" style={{ fontSize: 12 }}>{entry.account}</span>
                  <span style={{ flex: 1, fontSize: 12, color: 'var(--text-muted)' }}>
                    {entry.reason || 'Analyst approved'}
                  </span>
                  <span style={{ fontSize: 11, color: 'var(--green)' }}>
                    {entry.days_remaining}d remaining
                  </span>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    expires {new Date(entry.expires_at).toLocaleDateString()}
                  </span>
                  <button className="btn btn-danger btn-sm" onClick={() => revokeWhitelist(entry.account)}>
                    Revoke
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Search */}
        <div className="card" style={{ padding: '14px 20px' }}>
          <input
            className="search-input"
            style={{ width: '100%' }}
            placeholder="🔍 Search by account ID or entity name..."
            value={search}
            onChange={handleSearch}
          />
        </div>

        {/* Profile Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 14 }}>
          {profiles.map(p => {
            const riskMod = p.risk_modifier || 1.0
            const riskColor = riskMod < 0.7 ? '#10b981' : riskMod < 1.2 ? '#f59e0b' : '#ef4444'
            const wlEntry = whitelist.find(w => w.account === p.account)

            return (
              <div key={p.account} className="card" style={{ padding: '16px', cursor: 'pointer', position: 'relative' }}
                onClick={() => setSelected(selected?.account === p.account ? null : p)}>

                {/* Whitelist indicator */}
                {wlEntry && (
                  <div style={{
                    position: 'absolute', top: 10, right: 10, fontSize: 10,
                    background: 'rgba(16,185,129,0.15)', color: 'var(--green)',
                    padding: '2px 8px', borderRadius: 'var(--radius-full)', border: '1px solid rgba(16,185,129,0.3)'
                  }}>
                    ✓ Whitelisted
                  </div>
                )}

                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
                  <div style={{
                    width: 40, height: 40, borderRadius: '50%',
                    background: 'var(--bg-secondary)', display: 'flex', alignItems: 'center',
                    justifyContent: 'center', fontSize: 20, border: '1px solid var(--border)'
                  }}>
                    {ENTITY_ICON[p.entity_type] || '❓'}
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {p.entity_name || 'Unknown Entity'}
                    </div>
                    <div className="account-chip" style={{ fontSize: 10, marginTop: 2 }}>
                      {(p.account || '').slice(0, 14)}
                    </div>
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 12 }}>
                  {[
                    { label: 'Entity Type', value: p.entity_type },
                    { label: 'Transactions', value: (p.tx_count || 0).toLocaleString() },
                    { label: 'Avg Sent', value: `$${((p.avg_sent_usd || 0)).toLocaleString(undefined, { maximumFractionDigits: 0 })}` },
                    { label: 'Unique Recipients', value: p.unique_recipients || 0 },
                  ].map(item => (
                    <div key={item.label} style={{ background: 'var(--bg-secondary)', padding: '6px 10px', borderRadius: 'var(--radius-sm)' }}>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{item.label}</div>
                      <div style={{ color: 'var(--text-primary)', fontWeight: 500 }}>{item.value}</div>
                    </div>
                  ))}
                </div>

                {/* Risk modifier indicator */}
                <div style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Risk Modifier</span>
                  <div style={{ flex: 1, height: 4, background: 'rgba(255,255,255,0.07)', borderRadius: 2, overflow: 'hidden' }}>
                    <div style={{
                      width: `${Math.min(100, riskMod * 50)}%`, height: '100%',
                      background: riskColor, borderRadius: 2
                    }} />
                  </div>
                  <span style={{ fontSize: 11, color: riskColor, fontFamily: 'var(--font-mono)' }}>
                    ×{riskMod.toFixed(1)}
                  </span>
                </div>

                {p.historical_laundering_rate > 0 && (
                  <div style={{ marginTop: 8, fontSize: 11, color: '#f87171' }}>
                    ⚠ Historical laundering rate: {(p.historical_laundering_rate * 100).toFixed(1)}%
                  </div>
                )}

                {p.is_benign_keyword && (
                  <div style={{ marginTop: 4, fontSize: 11, color: 'var(--green)' }}>
                    ✓ Benign entity type detected (suppression active)
                  </div>
                )}
              </div>
            )
          })}
        </div>

        {profiles.length === 0 && !loading && (
          <div style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-muted)' }}>
            <div style={{ fontSize: 40 }}>👤</div>
            <div style={{ marginTop: 12 }}>
              {search ? 'No profiles match your search.' : 'Run training first to build customer profiles.'}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
