@app.get("/debris/catalog")
def get_debris_catalog(limit: int = 20):
    """
    Fetch real debris objects from Space-Track catalog.
    Returns TLE data for tracked debris in LEO.
    """
    try:
        # Query Space-Track for debris objects
        # NORAD CAT IDs for known debris clouds:
        # - Fengyun-1C: 39999 (and fragments)
        # - Cosmos-Iridium collision debris
        # - ISS vicinity objects
        
        debris_ids = [
            39999,  # FENGYUN 1C DEB
            22403,  # SL-16 DEB  
            25400,  # COSMOS 2251 DEB
            33591,  # IRIDIUM 33 DEB
            37756,  # BREEZE-M DEB
            40294,  # H-2A DEB
            41731,  # DELTA 2 DEB
            43947,  # CZ-4B DEB
        ]
        
        debris_list = []
        
        for debris_id in debris_ids[:limit]:
            try:
                tle_data = get_latest_tle(debris_id)
                if tle_data:
                    debris_list.append({
                        "id": str(debris_id),
                        "name": tle_data.get("name", f"DEBRIS-{debris_id}"),
                        "line1": tle_data["line1"],
                        "line2": tle_data["line2"]
                    })
            except Exception as e:
                print(f"Failed to fetch debris {debris_id}: {e}")
                continue
        
        return {"debris": debris_list, "count": len(debris_list)}
        
    except Exception as e:
        print(f"Debris catalog error: {e}")
        return {"debris": [], "count": 0, "error": str(e)}
