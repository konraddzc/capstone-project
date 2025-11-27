## PROJECT REQUIREMENTS

A customer has approached us with a requirement to create a data set that their Data Analysts and Data Scientists can work with.

The customer requires a robust ETL pipeline to integrate transaction data from a SQL database and demographic data from a CSV file. The pipeline must clean and standardise the data, remove invalid or incomplete records, and retain only active customers who have spent over $500. Additionally, it should enrich the dataset by calculating total customer spending and average transaction value per customer. This is so the company can target high-value customers with relevant marketing and rewards.  The final dataset must be stored in SQL and updated regularly for accurate analysis.

---
---

## PROJECT REQUIREMENTS AS AN EPIC

```text
As a THE CUSTOMER,
I want a robust ETL pipeline that integrates, cleans, standardises, and enriches transaction and demographic data from SQL and CSV sources, retaining only active customers who have spent over $500 and calculating total and average spend per customer,
So that high-value customers can be identified and targeted with marketing and rewards, using an up-to-date dataset stored in SQL for accurate analysis.
```

---
---

## EPIC 1

```text
As a Data Analyst/Scientist,
I want to be able to access the customer and transactions,
So that it can be transformed ready for analysis
```

---
---

## EPIC 2

```text
As a Data Analyst/Scientist,
I want to be able to have clean, standardised, enriched and aggregated data,
So that it can be analysed easier
```

---
---

## EPIC 3

```text
As a Data Analyst/Scientist,
I want to be able to have the extracted, transformed data in a single SQL table,
So that analysis can be done on high value customers
```

---
---

## EPIC 1 BREAKDOWN

```text
As a Data Analyst/Scientist,
I want to be able to access the customer and transactions,
So that it can be transformed ready for analysis
```

This can be broken down into 2 user stories:

### USER STORY 1

```text
As a Data Analyst/Scientist,
I want to be able to access the transaction data from the SQL database,
So that it can be transformed ready for analysis
```

#### USER STORY 1 ACCEPTANCE CRITERIA

- [x] The transaction data is extracted from the SQL database
- [x] Transaction data extraction is executed in less than 2 seconds
- [x] Extraction occurs without errors and data integrity is maintained (no missing or corrupted data)
- [x] Successful extractions are logged
- [x] Database connection errors are logged and handled gracefully
- [x] Database transaction errors are logged and handled gracefully
- [x] Transaction data is stored in a Pandas DataFrame for further processing
- [x] Tests are written to verify the transaction data extraction process

---

### USER STORY 2

```text
As a Data Analyst/Scientist,
I want to be able to access the customer data from the CSV file,
So that it can be transformed, ready for analysis
```

#### USER STORY 2 ACCEPTANCE CRITERIA

- [x] The customer data is extracted from the supplied CSV file
- [x] Customer data extraction is executed in less than 2 seconds
- [x] Extraction occurs without errors and data integrity is maintained (no missing or corrupted data)
- [x] Successful extractions are logged
- [x] CSV and file errors are logged and handled gracefully
- [x] Customer data is stored in a Pandas DataFrame for further processing
- [x] Tests are written to verify the customer data extraction process

---
---

## EPIC 2 BREAKDOWN

```text
As a Data Analyst/Scientist,
I want to be able to have clean, standardised, enriched and aggregated data,
So that it can be analysed easier
```

This can be broken down into 3 user stories:

### USER STORY 3

```text
As a Data Analyst/Scientist,
I want to be able to access cleaned, standardised transaction data,
So that it can be combined with the customer data and made available as a single table
```

> An acceptance criteria should be in place for this...

### USER STORY 4

```text
As a Data Analyst/Scientist,
I want to be able to access cleaned, standardised customer data,
So that it can be combined with the transaction data and made available as a single table
```

> An acceptance criteria should be in place for this...

###  USER STORY 5

```text
As a Data Analyst/Scientist,
I want to be able to access the combined, enriched and aggregated transaction and customer data,
So that it can be analysed easier
```

> An acceptance criteria should be in place for this...

---
---
