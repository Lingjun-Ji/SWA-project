import pandas as pd
from shiny import App, ui, render, reactive
import os
from loadjson import extract_orgs_from_json
import url_general_adapter as urlscrapper

filename = "C:/Users/KrisJ/Desktop/SWA_CODE/options.xlsx"

xls = pd.ExcelFile(filename)

def get_options(sheet_name):
    df = pd.read_excel(xls, sheet_name=sheet_name)
    df.columns = df.columns.str.strip()
    df["Value"] = df["Value"].astype(str)
    df["Label"] = df["Label"].astype(str)
    return dict(zip(df["Value"], df["Label"])), dict(zip(df["Value"], df["Value"]))

country_labels, country_values = get_options("Geolocation")
industry_labels, industry_values = get_options("Industry")
sdg_labels, sdg_values = get_options("SDG")
year_labels, year_values = get_options("Year")
doc_labels, doc_values = get_options("Document Type")
freq_labels, freq_values = get_options("Frequency")

json_dir = "C:/Users/KrisJ/Desktop/SWA_CODE/AUSTRALIA ANZSIC"
org_data = extract_orgs_from_json(json_dir)

def make_multiselect_with_placeholder(id, label, choices, placeholder):
    return ui.input_selectize(id, label, choices=choices, multiple=True, options={"placeholder": placeholder})

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.tags.h2("\U0001F33F SWA Sustainability Input Panel", style="color:#2E8B57; margin-bottom: 10px;"),
        make_multiselect_with_placeholder("country", "\U0001F310 Country", list(country_values.keys()), "Select a Country"),
        make_multiselect_with_placeholder("industry", "\U0001F3ED Industry (ANZSIC)", list(industry_values.keys()), "Select an Industry"),
        make_multiselect_with_placeholder("sdg", "\U0001F3AF SDG Goal", list(sdg_values.keys()), "Select a SDG Goal"),
        ui.input_select("year", "\U0001F4C5 Year", choices=["Select a Year"] + list(year_values.keys())),
        make_multiselect_with_placeholder("document_type", "\U0001F4C4 Document Type", list(doc_values.keys()), "Select Document Type"),
        ui.input_select("Frequency", "\U0001F504 Frequency", choices=["Select Frequency"] + list(freq_values.keys())),
        ui.input_action_button("submit", "\U0001F680 Submit", class_="btn btn-success mt-3"),
        ui.tags.div(ui.output_text("error_message"), style="color: red; font-weight: bold; margin-top: 10px;")
    ),

    ui.layout_columns(
        ui.card(
            ui.tags.h3("\u2705 Your Selections", style="color: #2c3e50;"),
            ui.output_text_verbatim("selection"),
            class_="p-3"
        ),
        ui.card(
            ui.tags.h3("\U0001F4CA Organization Results", style="color: #2c3e50;"),
            ui.output_text_verbatim("org_results"),
            class_="p-3"
        )
    ),

    ui.tags.footer(
        ui.tags.hr(),
        ui.tags.p("\U0001F30E Sustainable World Alliance • Powered by RNB Media", style="text-align:center; font-size: 0.9em; color: #666;")
    )
)

def server(input, output, session):
    submit_clicked = reactive.value(False)

    @output
    @render.text
    def error_message():
        if submit_clicked() and missing_fields():
            return f"⚠️ Please select: {', '.join(missing_fields())}."
        return ""

    @output
    @render.text
    def selection():
        if submit_clicked() and missing_fields():
            return f"Please select: {', '.join(missing_fields())}."
        return (
            f"\U0001F310 Country: {', '.join([country_labels.get(c, c) for c in input.country()])}\n"
            f"\U0001F3ED Industry: {', '.join([industry_labels.get(i, i) for i in input.industry()])}\n"
            f"\U0001F3AF SDG Goal: {', '.join([sdg_labels.get(s, s) for s in input.sdg()])}\n"
            f"\U0001F4C5 Year: {year_labels.get(input.year(), 'Select a Year')}\n"
            f"\U0001F4C4 Document Type: {', '.join([doc_labels.get(d, d) for d in input.document_type()])}\n"
            f"\U0001F504 Frequency: {freq_labels.get(input['Frequency'](), 'Select Frequency')}\n"
        )

    @output
    @render.text
    def org_results():
        if not submit_clicked():
            return ""

        selected_divisions = input.industry()
        matched_orgs = [f"• {org['organisation_name']}" for org in org_data if org.get("division") in selected_divisions]

        if not matched_orgs:
            return "No matching organizations found."

        return "\n".join(matched_orgs)

    @reactive.effect
    def _():
        if input.submit():
            submit_clicked.set(True)
            if missing_fields():
                return

            user_inputs = {
                "country": input.country(),
                "industry": input.industry(),
                "sdg_raw": input.sdg(),
                "sdg_labels": [sdg_labels.get(s, s) for s in input.sdg()],
                "year": input.year(),
                "doc_type_raw": input.document_type(),
                "doc_labels": [doc_labels.get(d, d) for d in input.document_type()]
            }
            print("DEBUG sdg_labels from UI =", user_inputs["sdg_labels"])
            selected_divisions = input.industry()
            matched_orgs = [org for org in org_data if org.get("division") in selected_divisions]
            urlscrapper.generate_urls(user_inputs, matched_orgs)

    @reactive.calc
    def missing_fields():
        fields = []
        if not input.country(): fields.append("Country")
        if not input.industry(): fields.append("Industry")
        if not input.sdg(): fields.append("SDG Goal")
        if input.year() == "Select a Year": fields.append("Year")
        if not input.document_type(): fields.append("Document Type")
        if input.Frequency() == "Select Frequency": fields.append("Frequency")
        return fields

app = App(app_ui, server)

if __name__ == "__main__":
    from shiny import run_app
    run_app(app, port=8000, host="127.0.0.1")

