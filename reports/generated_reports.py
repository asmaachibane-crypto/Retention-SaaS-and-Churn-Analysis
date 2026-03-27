if st.button("Generate Report"):

    df.to_csv("reports/generated_reports/churn_report.csv", index=False)

    st.success("Report generated successfully")
    