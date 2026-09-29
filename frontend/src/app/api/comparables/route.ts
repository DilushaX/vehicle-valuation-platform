import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

interface ComparableRequest {
  category: string;
  brand: string;
  model: string;
  manufacture_year: number;
  mileage: number;
  engine_cc: number;
  fuel_type: string;
  transmission: string;
  district: string;
  condition: string;
  top_k?: number;
  weights?: {
    brand?: number;
    model?: number;
    year?: number;
    mileage?: number;
    engine_cc?: number;
    transmission?: number;
    fuel_type?: number;
    district?: number;
    condition?: number;
  };
}

export async function POST(req: NextRequest) {
  try {
    const body: ComparableRequest = await req.json();
    const projectRoot = path.resolve(process.cwd(), "..");
    const csvPath = path.join(projectRoot, "data", "analysis", "ml", "ml_dataset.csv");

    if (!fs.existsSync(csvPath)) {
      return NextResponse.json({ status: "error", message: "Dataset not found" }, { status: 404 });
    }

    const content = fs.readFileSync(csvPath, "utf-8");
    const lines = content.split("\n").filter((l) => l.trim().length > 0);
    const headers = lines[0].split(",");

    const rows = lines.slice(1).map((line) => {
      const vals = line.split(",");
      const obj: Record<string, any> = {};
      headers.forEach((h, i) => {
        obj[h.trim()] = vals[i] ? vals[i].trim() : "";
      });
      return {
        listing_id: obj.listing_id,
        category: obj.category,
        brand: obj.brand,
        model: obj.model,
        manufacture_year: parseInt(obj.manufacture_year, 10) || 0,
        mileage: parseFloat(obj.mileage) || 0,
        engine_cc: parseFloat(obj.engine_cc) || 0,
        fuel_type: obj.fuel_type,
        transmission: obj.transmission,
        district: obj.district,
        condition: obj.condition,
        asking_price: parseFloat(obj.asking_price) || 0,
      };
    });

    // Filter to category if provided
    const filtered = body.category
      ? rows.filter((r) => r.category.toLowerCase() === body.category.toLowerCase())
      : rows;

    // Weights
    const w = {
      brand: body.weights?.brand ?? 0.25,
      model: body.weights?.model ?? 0.25,
      year: body.weights?.year ?? 0.14,
      mileage: body.weights?.mileage ?? 0.10,
      engine_cc: body.weights?.engine_cc ?? 0.08,
      transmission: body.weights?.transmission ?? 0.06,
      fuel_type: body.weights?.fuel_type ?? 0.04,
      district: body.weights?.district ?? 0.04,
      condition: body.weights?.condition ?? 0.04,
    };

    const totalWeight = Object.values(w).reduce((a, b) => a + b, 0) || 1.0;

    const scored = filtered.map((item) => {
      let score = 0;

      // Brand match
      if (item.brand.toLowerCase() === body.brand.toLowerCase()) {
        score += w.brand;
      }

      // Model match
      if (item.model.toLowerCase() === body.model.toLowerCase()) {
        score += w.model;
      } else if (item.model.toLowerCase().includes(body.model.toLowerCase()) || body.model.toLowerCase().includes(item.model.toLowerCase())) {
        score += w.model * 0.6;
      }

      // Year similarity (decay by diff)
      const yearDiff = Math.abs(item.manufacture_year - (body.manufacture_year || 2016));
      score += w.year * Math.max(0, 1 - yearDiff / 12);

      // Mileage similarity
      const mileageDiff = Math.abs(item.mileage - (body.mileage || 80000));
      score += w.mileage * Math.max(0, 1 - mileageDiff / 150000);

      // Engine CC similarity
      const ccDiff = Math.abs(item.engine_cc - (body.engine_cc || 1500));
      score += w.engine_cc * Math.max(0, 1 - ccDiff / 2000);

      // Transmission
      if (item.transmission.toLowerCase() === (body.transmission || "").toLowerCase()) {
        score += w.transmission;
      }

      // Fuel type
      if (item.fuel_type.toLowerCase() === (body.fuel_type || "").toLowerCase()) {
        score += w.fuel_type;
      }

      // District
      if (item.district.toLowerCase() === (body.district || "").toLowerCase()) {
        score += w.district;
      }

      // Condition
      if (item.condition.toLowerCase() === (body.condition || "").toLowerCase()) {
        score += w.condition;
      }

      const normalized = Math.min(1.0, Math.max(0.0, score / totalWeight));
      return {
        ...item,
        similarity_score: parseFloat(normalized.toFixed(4)),
        similarity_percentage: Math.round(normalized * 100),
      };
    });

    scored.sort((a, b) => b.similarity_score - a.similarity_score);
    const topK = body.top_k || 6;
    const results = scored.slice(0, topK);

    // Distribution summary
    const prices = results.map((r) => r.asking_price).filter((p) => p > 0);
    prices.sort((a, b) => a - b);
    const minPrice = prices.length ? prices[0] : 0;
    const maxPrice = prices.length ? prices[prices.length - 1] : 0;
    const medianPrice = prices.length
      ? prices[Math.floor(prices.length / 2)]
      : 0;
    const meanPrice = prices.length
      ? Math.round(prices.reduce((a, b) => a + b, 0) / prices.length)
      : 0;

    return NextResponse.json({
      status: "ok",
      comparables: results,
      total_matches: scored.length,
      market_summary: {
        min: minPrice,
        median: medianPrice,
        mean: meanPrice,
        max: maxPrice,
        count: results.length,
      },
      weights_applied: w,
    });
  } catch (err: any) {
    return NextResponse.json(
      { status: "error", message: err.message || "Failed to search comparables" },
      { status: 500 }
    );
  }
}
