import os
import csv
from dotenv import load_dotenv

# Load .env first so that POSTGRES_DATABASE_URL is set before importing db
load_dotenv()

from sqlalchemy.schema import CreateTable
from db import engine, SessionLocal
from models import User, SongMetadata, UserLikedSong

output_dir = r"c:\Users\Agrannya Singh\.antigravity\tunetrace\samsung_prism_phase_1_submission"
sql_file = os.path.join(output_dir, "table_definitions.sql")
csv_file = os.path.join(output_dir, "song_metadata_export.csv")

# 1. Export SQL Definitions
with open(sql_file, "w", encoding="utf-8") as f:
    f.write("-- Database Schema Export\n\n")
    for model in [User, SongMetadata, UserLikedSong]:
        try:
            create_stmt = str(CreateTable(model.__table__).compile(engine))
            f.write(create_stmt + ";\n\n")
        except Exception as e:
            f.write(f"-- Failed to generate schema for {model.__tablename__}: {e}\n\n")

print(f"Generated SQL definitions at {sql_file}")

# 2. Export 100 songs to CSV
db = SessionLocal()
try:
    songs = db.query(SongMetadata).limit(100).all()
    with open(csv_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "video_id", "title", "artist", "genre", "tags", "enriched", "embedding_preview"])
        for s in songs:
            if s.embedding is not None:
                emb_str = str(s.embedding)
                if len(emb_str) > 50:
                    emb_preview = emb_str[:50] + "...]"
                else:
                    emb_preview = emb_str
            else:
                emb_preview = "NULL"
                
            writer.writerow([s.id, s.video_id, s.title, s.artist, s.genre, s.tags, s.enriched, emb_preview])
    print(f"Exported {len(songs)} rows to {csv_file}")
except Exception as e:
    print(f"Failed to export CSV: {e}")
finally:
    db.close()
