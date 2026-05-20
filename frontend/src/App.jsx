import { useState, useEffect } from "react"
import axios from "axios"
import {
  BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from "recharts"

const API = "http://127.0.0.1:8000"

// ── valeurs par défaut d'un patient test ──────────────────
const PATIENT_DEFAUT = {
  age: 55, sex: 1, cp: 0, trestbps: 130,
  chol: 250, fbs: 0, restecg: 0, thalach: 150,
  exang: 0, oldpeak: 1.5, slope: 1, ca: 0, thal: 2,
}

const LABELS = {
  age: "Âge", sex: "Sexe (1=H, 0=F)", cp: "Douleur thoracique (0-3)",
  trestbps: "Pression artérielle", chol: "Cholestérol",
  fbs: "Glycémie > 120 (0/1)", restecg: "ECG repos (0-2)",
  thalach: "Fréq. cardiaque max", exang: "Angine effort (0/1)",
  oldpeak: "Dépression ST", slope: "Pente ST (0-2)",
  ca: "Vaisseaux colorés (0-4)", thal: "Thalassémie (0-3)",
}

const MODELES = [
  "Logistic_Regression", "SVM",
  "Random_Forest", "KNN", "Neural_Network",
  "AdaBoost", "XGBoost"   // ← ces 2 ajoutés
]

export default function App() {
  const [patient, setPatient]         = useState(PATIENT_DEFAUT)
  const [modeleChoisi, setModele]     = useState("Logistic_Regression")
  const [resultat, setResultat]       = useState(null)
  const [scores, setScores]           = useState([])
  const [loading, setLoading]         = useState(false)
  const [onglet, setOnglet]           = useState("predict")

  // Charger les scores au démarrage
  useEffect(() => {
    axios.get(`${API}/results`)
      .then(r => setScores(r.data))
      .catch(() => {})
  }, [])

  const handleChange = (e) => {
    const { name, value } = e.target
    setPatient(p => ({ ...p, [name]: parseFloat(value) }))
  }

  const predire = async () => {
    setLoading(true)
    setResultat(null)
    try {
      const r = await axios.post(`${API}/predict/${modeleChoisi}`, patient)
      setResultat(r.data)
    } catch {
      setResultat({ erreur: "Erreur de connexion à l'API" })
    }
    setLoading(false)
  }

  return (
    <div className="min-h-screen bg-gray-50 font-sans">

      {/* ── HEADER ── */}
      <header className="bg-red-600 text-white px-8 py-5 shadow-md">
        <h1 className="text-2xl font-bold">
          🫀 Heart Disease Classifier
        </h1>
        <p className="text-red-100 text-sm mt-1">
          Identification des patients à risque cardiaque
        </p>
      </header>

      {/* ── ONGLETS ── */}
      <div className="flex gap-2 px-8 pt-6">
        {[
          { id: "predict",  label: "🔬 Prédiction" },
          { id: "compare",  label: "📊 Comparaison modèles" },
        ].map(o => (
          <button key={o.id} onClick={() => setOnglet(o.id)}
            className={`px-5 py-2 rounded-full text-sm font-medium transition
              ${onglet === o.id
                ? "bg-red-600 text-white shadow"
                : "bg-white text-gray-600 border hover:bg-gray-100"}`}>
            {o.label}
          </button>
        ))}
      </div>

      <main className="px-8 py-6 max-w-6xl mx-auto">

        {/* ══════════════════════════════════════════
            ONGLET 1 — PRÉDICTION
        ══════════════════════════════════════════ */}
        {onglet === "predict" && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

            {/* Formulaire patient */}
            <div className="bg-white rounded-2xl shadow p-6">
              <h2 className="text-lg font-semibold text-gray-800 mb-4">
                📋 Données du patient
              </h2>

              <div className="grid grid-cols-2 gap-3">
                {Object.entries(LABELS).map(([key, label]) => (
                  <div key={key}>
                    <label className="text-xs text-gray-500 block mb-1">
                      {label}
                    </label>
                    <input
                      type="number"
                      name={key}
                      value={patient[key]}
                      onChange={handleChange}
                      step={key === "oldpeak" ? "0.1" : "1"}
                      className="w-full border rounded-lg px-3 py-2 text-sm
                                 focus:outline-none focus:ring-2 focus:ring-red-300"
                    />
                  </div>
                ))}
              </div>

              {/* Choix du modèle */}
              <div className="mt-4">
                <label className="text-xs text-gray-500 block mb-1">
                  Algorithme ML
                </label>
                <select
                  value={modeleChoisi}
                  onChange={e => setModele(e.target.value)}
                  className="w-full border rounded-lg px-3 py-2 text-sm
                             focus:outline-none focus:ring-2 focus:ring-red-300">
                  {MODELES.map(m => (
                    <option key={m} value={m}>
                      {m.replace(/_/g, " ")}
                    </option>
                  ))}
                </select>
              </div>

              <button
                onClick={predire}
                disabled={loading}
                className="mt-5 w-full bg-red-600 hover:bg-red-700 disabled:bg-red-300
                           text-white font-semibold py-3 rounded-xl transition text-sm">
                {loading ? "Analyse en cours..." : "🔍 Analyser le patient"}
              </button>
            </div>

            {/* Résultat */}
            <div className="bg-white rounded-2xl shadow p-6 flex flex-col justify-center">
              <h2 className="text-lg font-semibold text-gray-800 mb-4">
                📊 Résultat du diagnostic
              </h2>

              {!resultat && (
                <div className="text-center text-gray-400 py-16">
                  <p className="text-5xl mb-4">🫀</p>
                  <p className="text-sm">
                    Remplis les données et clique sur Analyser
                  </p>
                </div>
              )}

              {resultat?.erreur && (
                <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-red-700">
                  ⚠️ {resultat.erreur}
                </div>
              )}

              {resultat && !resultat.erreur && (
                <div className="space-y-4">

                  {/* Verdict */}
                  <div className={`rounded-2xl p-6 text-center
                    ${resultat.prediction === 1
                      ? "bg-red-50 border-2 border-red-300"
                      : "bg-green-50 border-2 border-green-300"}`}>
                    <p className="text-5xl mb-2">
                      {resultat.prediction === 1 ? "🔴" : "🟢"}
                    </p>
                    <p className={`text-2xl font-bold
                      ${resultat.prediction === 1
                        ? "text-red-700" : "text-green-700"}`}>
                      {resultat.diagnostic}
                    </p>
                    <p className="text-sm text-gray-500 mt-1">
                      Modèle : {resultat.modele.replace(/_/g, " ")}
                    </p>
                  </div>

                  {/* Probabilités */}
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-green-50 rounded-xl p-4 text-center">
                      <p className="text-xs text-gray-500 mb-1">Probabilité Sain</p>
                      <p className="text-2xl font-bold text-green-600">
                        {resultat.probabilite_sain}%
                      </p>
                    </div>
                    <div className="bg-red-50 rounded-xl p-4 text-center">
                      <p className="text-xs text-gray-500 mb-1">Probabilité Malade</p>
                      <p className="text-2xl font-bold text-red-600">
                        {resultat.probabilite_malade}%
                      </p>
                    </div>
                  </div>

                  {/* Barre de probabilité */}
                  <div>
                    <p className="text-xs text-gray-500 mb-1">Risque cardiaque</p>
                    <div className="w-full bg-gray-100 rounded-full h-4">
                      <div
                        className="h-4 rounded-full transition-all duration-700"
                        style={{
                          width: `${resultat.probabilite_malade}%`,
                          backgroundColor: resultat.probabilite_malade > 50
                            ? "#dc2626" : "#16a34a"
                        }}
                      />
                    </div>
                    <p className="text-xs text-gray-400 mt-1 text-right">
                      {resultat.probabilite_malade}% de risque
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ══════════════════════════════════════════
            ONGLET 2 — COMPARAISON MODÈLES
        ══════════════════════════════════════════ */}
        {onglet === "compare" && (
          <div className="bg-white rounded-2xl shadow p-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-6">
              📊 Comparaison des modèles ML
            </h2>

            {scores.length === 0 ? (
              <p className="text-gray-400 text-center py-10">
                Chargement...
              </p>
            ) : (
              <>
                {/* Graphique */}
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={scores} margin={{ top: 10, right: 30, left: 0, bottom: 60 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0"/>
                    <XAxis dataKey="Modèle"
                      angle={-20} textAnchor="end"
                      tick={{ fontSize: 12 }}/>
                    <YAxis domain={[60, 100]} unit="%" tick={{ fontSize: 12 }}/>
                    <Tooltip formatter={(v) => `${v}%`}/>
                    <Legend wrapperStyle={{ paddingTop: 20 }}/>
                    <Bar dataKey="Accuracy" fill="#2196F3" radius={[4,4,0,0]}/>
                    <Bar dataKey="F1-Score" fill="#4CAF50" radius={[4,4,0,0]}/>
                    <Bar dataKey="AUC-ROC"  fill="#FF9800" radius={[4,4,0,0]}/>
                  </BarChart>
                </ResponsiveContainer>

                {/* Tableau */}
                <table className="w-full mt-6 text-sm">
                  <thead>
                    <tr className="bg-gray-50 text-gray-600">
                      <th className="text-left px-4 py-3 rounded-l-lg">Modèle</th>
                      <th className="px-4 py-3 text-center">Accuracy</th>
                      <th className="px-4 py-3 text-center">F1-Score</th>
                      <th className="px-4 py-3 text-center rounded-r-lg">AUC-ROC</th>
                    </tr>
                  </thead>
                  <tbody>
                    {scores.map((s, i) => (
                      <tr key={i}
                        className={`border-t ${i === 0
                          ? "bg-red-50 font-semibold" : "hover:bg-gray-50"}`}>
                        <td className="px-4 py-3">
                          {i === 0 && "🏆 "}{s["Modèle"]}
                        </td>
                        <td className="px-4 py-3 text-center text-blue-600">
                          {s["Accuracy"]}%
                        </td>
                        <td className="px-4 py-3 text-center text-green-600">
                          {s["F1-Score"]}%
                        </td>
                        <td className="px-4 py-3 text-center text-orange-500">
                          {s["AUC-ROC"]}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            )}
          </div>
        )}
      </main>
    </div>
  )
}