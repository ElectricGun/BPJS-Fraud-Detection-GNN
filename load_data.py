from py2neo import Graph, Node, Relationship
import pandas as pd
import config

# 1. Koneksi ke Neo4j
graph = Graph(config.url, auth=(config.uname, config.pw))

# 2. Load dataset
df = pd.read_csv("claims_synthetic_1200.csv")
df.columns = df.columns.str.strip().str.lower()
print("Jumlah data:", len(df))

# 3. Bersihkan database
graph.run("MATCH (n) DETACH DELETE n;")

# 4. Masukkan node dan relasi
for _, row in df.iterrows():
    p = Node("Patient", id_pasien=row["id pasien"])
    c = Node(
        "Claim",
        id_klaim=row["id klaim"],
        diagnosis=row.get("diagnosis utama"),
        prosedur=row.get("prosedur"),
        tarif_seharusnya=row.get("tarif seharusnya (rp)"),
        tarif_diklaim=row.get("tarif diklaim (rp)"),
        jenis_pelayanan=row.get("jenis pelayanan"),
        lama_rawat=row.get("lama rawat (hari)"),
        kelas_rawat=row.get("kelas rawat"),
        status_klaim=row.get("status klaim"),
        jenis_fraud=row.get("jenis fraud (jika terbukti)"),
        catatan=row.get("catatan"),
        is_fraud=row.get("is_fraud")
    )

    graph.merge(p, "Patient", "id_pasien")
    graph.merge(c, "Claim", "id_klaim")
    graph.merge(Relationship(p, "MADE_CLAIM", c))

print("✅ Data klaim berhasil dimuat ke Neo4j!")