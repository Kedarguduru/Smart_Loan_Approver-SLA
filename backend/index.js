// backend/index.js  — Node 22 compatible (NO TensorFlow). Uses Decision Tree via ml-cart.
const express = require("express");
const cors = require("cors");
const fs = require("fs");
const path = require("path");
const csv = require("csv-parser");
const { DecisionTreeClassifier } = require("ml-cart");

const app = express();
app.use(cors());
app.use(express.json());

// Paths & columns
const CSV_FILE = path.join(__dirname, "data", "M2 T2.csv");
const MODEL_DIR = path.join(__dirname, "ml_model");
const MODEL_FILE = path.join(MODEL_DIR, "tree.json");
const META_FILE = path.join(MODEL_DIR, "meta.json");

// Final feature set (exactly these 3)
const FEATURE_COLUMNS = ["Credit_Score", "Income", "Loan_Amount(s)"];
const TARGET_COLUMN = "Loan_Approved";

// Helpers
function labelTo01(v) {
    if (v === undefined || v === null) return null;
    const s = String(v).trim().toLowerCase();
    if (["yes", "y", "true", "1", "approved", "approve"].includes(s)) return 1;
    if (["no", "n", "false", "0", "rejected", "reject"].includes(s)) return 0;
    return null;
}

function loadCSVRows() {
    return new Promise((resolve, reject) => {
        const rows = [];
        fs.createReadStream(CSV_FILE)
            .pipe(csv())
            .on("data", (row) => rows.push(row))
            .on("end", () => resolve(rows))
            .on("error", reject);
    });
}

app.get("/", (_req, res) => {
    res.send("✅ Loan Decision Tree API (Features: Credit_Score, Income, Loan_Amount(s))");
});

// Train model from CSV
app.post("/train", async (_req, res) => {
    try {
        const rows = await loadCSVRows();
        if (!rows.length) throw new Error("CSV is empty.");
        const headers = Object.keys(rows[0] || {});
        for (const needed of [...FEATURE_COLUMNS, TARGET_COLUMN]) {
            if (!headers.includes(needed)) {
                throw new Error(`CSV missing required column: ${needed}`);
            }
        }

        const X = [];
        const y = [];
        let skipped = 0;

        for (const r of rows) {
            const label = labelTo01(r[TARGET_COLUMN]);
            const vec = FEATURE_COLUMNS.map((f) => Number(r[f]));
            if (label === null || vec.some((v) => Number.isNaN(v))) {
                skipped++;
                continue;
            }
            X.push(vec);
            y.push(label);
        }

        if (X.length < 2) throw new Error(`Not enough valid rows to train (got ${X.length}, skipped ${skipped}).`);

        const clf = new DecisionTreeClassifier(); // CART classifier
        clf.train(X, y);

        fs.mkdirSync(MODEL_DIR, { recursive: true });
        fs.writeFileSync(MODEL_FILE, JSON.stringify(clf.toJSON(), null, 2));
        fs.writeFileSync(
            META_FILE,
            JSON.stringify({ FEATURES: FEATURE_COLUMNS, TARGET: TARGET_COLUMN, samplesUsed: X.length, skipped }, null, 2)
        );

        res.json({
            ok: true,
            message: "Model trained & saved.",
            features: FEATURE_COLUMNS,
            target: TARGET_COLUMN,
            samples_used: X.length,
            rows_skipped: skipped,
        });
    } catch (err) {
        console.error(err);
        res.status(500).json({ ok: false, error: err.message });
    }
});

// Predict with a trained model
// Body: { "record": { "Credit_Score": 720, "Income": 55000, "Loan_Amount(s)": 20000 } }
app.post("/predict", async (req, res) => {
    try {
        if (!fs.existsSync(MODEL_FILE) || !fs.existsSync(META_FILE)) {
            return res.status(400).json({ ok: false, error: "Model not trained. Run POST /train first." });
        }
        const treeJSON = JSON.parse(fs.readFileSync(MODEL_FILE, "utf-8"));
        const meta = JSON.parse(fs.readFileSync(META_FILE, "utf-8"));
        const clf = DecisionTreeClassifier.load(treeJSON);

        const rec = req.body?.record || {};
        const vec = meta.FEATURES.map((f) => Number(rec[f]));
        if (vec.length !== meta.FEATURES.length || vec.some((v) => Number.isNaN(v))) {
            return res.status(400).json({
                ok: false,
                error: `Missing or invalid numeric inputs. Required: ${meta.FEATURES.join(", ")}`,
            });
        }

        const pred = clf.predict([vec])[0]; // 0 or 1
        res.json({
            ok: true,
            features_used: meta.FEATURES,
            input_vector: vec,
            prediction: pred === 1 ? "Approved ✅" : "Rejected ❌",
            label: pred,
        });
    } catch (err) {
        console.error(err);
        res.status(500).json({ ok: false, error: err.message });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🚀 API running at http://localhost:${PORT}`));
