import { useApp } from '../App'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import { useState, useEffect } from 'react'

const SEVERITY_COLORS = {
  CRITICAL: '#ef4444',
  HIGH: '#f97316',
  MEDIUM: '#f59e0b',
  LOW: '#10b981',
}

const PATTERN_COLORS = {
  'FAN-OUT': '#34d399', 'FAN-IN': '#f87171', 'CYCLE': '#a78bfa',
  'STACK': '#60a5fa', 'RANDOM': '#fb923c', 'BIPARTITE': '#fbbf24',
  'GATHER-SCATTER': '#e879f9', 'SCATTER-GATHER': '#2dd4bf',
}

export default function Dashboard() {
  const { alerts, stats, connected } = useApp()

  // Build pattern distribution
  const patternCounts = {}
  alerts.forEach(a => {
    const p = a.detected_pattern || 'UNKNOWN'
    patternCounts[p] = (patternCounts[p] || 0) + 1
  })
  const patternData = Object.entries(patternCounts)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 8)

  // Build timeline data (last 20 alerts)
  const timelineData = alerts.slice(0, 40).reverse().map((a, i) => ({
    i,
    score: Math.round((a.final_score || 0) * 100),
    severity: a.severity,
  }))

  // Severity distribution
  const sevCounts = { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0 }
  alerts.forEach(a => { if (sevCounts[a.severity] !== undefined) sevCounts[a.severity]++ })
  const sevData = Object.entries(sevCounts).map(([name, value]) => ({ name, value }))

  return (
    <div>
      <div className="page-header">
        <div>
          <div className="page-title">AML Detection Dashboard</div>
          <div className="page-subtitle">Real-time money laundering monitoring</div>
        </div>
        <div className="flex-center gap-2">
          <div className="live-indicator">
            <div className="live-dot" />
            LIVE
          </div>
          <span className="text-muted" style={{ fontSize: 12 }}>
            {alerts.length} events tracked
          </span>
        </div>
      </div>

      <div className="page-body">
        {/* KPI Cards */}
        <div className="kpi-grid">
          <div className="kpi-card blue">
            <div className="kpi-icon">🔍</div>
            <div className="kpi-value">{stats.total.toLocaleString()}</div>
            <div className="kpi-label">Total Alerts</div>
          </div>
          <div className="kpi-card red">
            <div className="kpi-icon">⚠️</div>
            <div className="kpi-value">{stats.pending}</div>
            <div className="kpi-label">Pending Review</div>
          </div>
          <div className="kpi-card orange">
            <div className="kpi-icon">🔥</div>
            <div className="kpi-value">{stats.critical}</div>
            <div className="kpi-label">Critical Severity</div>
          </div>
          <div className="kpi-card purple">
            <div className="kpi-icon">↩️</div>
            <div className="kpi-value">{stats.retroactive}</div>
            <div className="kpi-label">Retroactive Flags</div>
          </div>
          <div className="kpi-card green">
            <div className="kpi-icon">✅</div>
            <div className="kpi-value">{stats.approved}</div>
            <div className="kpi-label">Approved</div>
          </div>
          <div className="kpi-card yellow">
            <div className="kpi-icon">❌</div>
            <div className="kpi-value">{stats.rejected}</div>
            <div className="kpi-label">Rejected</div>
          </div>
        </div>

        {/* Charts Row */}
        <div className="grid-2">
          {/* Score Timeline */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Alert Score Timeline</span>
              <span className="text-muted" style={{ fontSize: 11 }}>Last 40 alerts</span>
            </div>
            <ResponsiveContainer width="100%" height={180}>
              <AreaChart data={timelineData}>
                <defs>
                  <linearGradient id="scoreGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="i" hide />
                <YAxis domain={[0, 100]} stroke="#4a5568" tick={{ fill: '#4a5568', fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ background: '#111827', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 8 }}
                  labelStyle={{ color: '#8b9dc3' }}
                  itemStyle={{ color: '#3b82f6' }}
                />
                <Area type="monotone" dataKey="score" stroke="#3b82f6" fill="url(#scoreGrad)" strokeWidth={2} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Pattern Distribution */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Pattern Distribution</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
              <PieChart width={140} height={140}>
                <Pie data={patternData} cx={65} cy={65} innerRadius={35} outerRadius={60}
                  dataKey="value" stroke="none">
                  {patternData.map((entry, i) => (
                    <Cell key={i} fill={PATTERN_COLORS[entry.name] || '#6b7280'} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ background: '#111827', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 8, fontSize: 12 }} />
              </PieChart>
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 6 }}>
                {patternData.slice(0, 6).map(p => (
                  <div key={p.name} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}>
                    <div style={{ width: 8, height: 8, borderRadius: '50%', background: PATTERN_COLORS[p.name] || '#6b7280', flexShrink: 0 }} />
                    <span style={{ color: 'var(--text-secondary)', flex: 1 }}>{p.name}</span>
                    <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{p.value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Recent Alerts */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Recent Alerts</span>
            <button className="btn btn-ghost btn-sm" onClick={() => window.location.href = '/alerts'}>
              View All →
            </button>
          </div>
          <div className="alerts-table-container">
            <table className="alerts-table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Pattern</th>
                  <th>From → To</th>
                  <th>Amount</th>
                  <th>Score</th>
                  <th>Severity</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {alerts.slice(0, 10).map(alert => (
                  <tr key={alert.id}>
                    <td className="mono" style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                      {alert.timestamp ? new Date(alert.timestamp).toLocaleTimeString() : '--'}
                    </td>
                    <td>
                      <span className={`pattern-badge ${alert.detected_pattern}`}>
                        {alert.detected_pattern || 'UNKNOWN'}
                      </span>
                      {alert.retroactive && <span className="retro-tag" style={{ marginLeft: 4 }}>↩ RETRO</span>}
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontSize: 11 }}>
                      <span className="account-chip">{(alert.from_account || '').slice(0, 10)}</span>
                      <span style={{ margin: '0 4px', color: 'var(--text-muted)' }}>→</span>
                      <span className="account-chip">{(alert.to_account || '').slice(0, 10)}</span>
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                      ${(alert.amount_usd || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </td>
                    <td>
                      <div className="score-bar-wrap">
                        <div className="score-bar">
                          <div className="score-fill" style={{
                            width: `${Math.round((alert.final_score || 0) * 100)}%`,
                            background: (alert.final_score || 0) > 0.7 ? 'var(--red)' :
                              (alert.final_score || 0) > 0.5 ? 'var(--yellow)' : 'var(--green)'
                          }} />
                        </div>
                        <span className="score-text">{Math.round((alert.final_score || 0) * 100)}%</span>
                      </div>
                    </td>
                    <td><span className={`severity-badge ${alert.severity}`}>{alert.severity}</span></td>
                    <td><span className={`status-badge ${alert.status}`}>{alert.status}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
            {alerts.length === 0 && (
              <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>
                <div style={{ fontSize: 32 }}>🔍</div>
                <div style={{ marginTop: 8 }}>No alerts yet. Start the transaction simulator.</div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
