import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET() {
  try {
    const projectRoot = path.resolve(process.cwd(), "..");
    const tablesDir = path.join(projectRoot, "data", "analysis", "tables");

    const readJson = (filename: string, fallback: any = null) => {
      try {
        const filePath = path.join(tablesDir, filename);
        if (fs.existsSync(filePath)) {
          return JSON.parse(fs.readFileSync(filePath, "utf-8"));
        }
      } catch (err) {
        console.error(`Failed to read ${filename}:`, err);
      }
      return fallback;
    };

    const overview = readJson("dataset_overview.json", {
      total_listings: 171,
      total_vehicles: 171,
      ml_eligible_listings: 113,
      ml_eligibility_rate_pct: 66.08,
      distinct_categories: 8,
      distinct_brands: 33,
      distinct_models: 112,
    });

    const categories = readJson("category_summary.json", []);
    const brands = readJson("brand_summary.json", []);
    const models = readJson("model_summary.json", []);
    const districts = readJson("district_summary.json", []);
    const fuel = readJson("fuel_summary.json", []);
    const transmission = readJson("transmission_summary.json", []);
    const numerical = readJson("numerical_summary.json", []);
    const correlations = readJson("correlations.json", {});
    const outliers = readJson("flagged_outliers.json", []);

    return NextResponse.json({
      status: "ok",
      overview,
      categories,
      brands,
      models,
      districts,
      fuel,
      transmission,
      numerical,
      correlations,
      outliers,
    });
  } catch (error) {
    return NextResponse.json(
      { status: "error", message: "Failed to load market analytics" },
      { status: 500 }
    );
  }
}
