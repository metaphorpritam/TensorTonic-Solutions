# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# Pritam Sarkar's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/pritamsba2027.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Implement Adam Optimizer Step | Implement one vectorized Adam optimizer step in NumPy with first and second moments, bias correction, and elementwise parameter updates. | https://www.tensortonic.com/problems/adam-optimizer |
| Implement AdamW (Decoupled Weight Decay) | Implement one AdamW optimizer step in NumPy with first and second moments plus decoupled weight decay. | https://www.tensortonic.com/problems/adamw-optimizer |
| Binomial Probability Mass Function | Compute binomial probability mass and cumulative probabilities from trial count, success probability, and outcome. | https://www.tensortonic.com/problems/binomial-pmf-cdf |
| Implement Cosine Similarity | Compute cosine similarity between NumPy vectors with dot products, Euclidean norms, and zero-vector handling. | https://www.tensortonic.com/problems/cosine-similarity |
| Compute Covariance Matrix | Compute a sample covariance matrix from centered observations, preserving feature-to-feature relationships. | https://www.tensortonic.com/problems/covariance-matrix |
| Implement Dot Product | Implement the dot product of equal-length numeric vectors by summing element-wise products without library shortcuts. | https://www.tensortonic.com/problems/dot-product |
| Implement Dropout (Training Mode) | Implement training-mode dropout in NumPy with random masking and inverted scaling of retained activations. | https://www.tensortonic.com/problems/dropout-training |
| Implement Euclidean Distance | Compute Euclidean distance between equal-length NumPy vectors as the square root of summed squared differences. | https://www.tensortonic.com/problems/euclidean-distance |
| Implement Gradient Descent for a 1D Quadratic | Optimize a one-dimensional quadratic with iterative gradient descent and return the parameter trajectory. | https://www.tensortonic.com/problems/gradient-descent-quadratic |
| Build a Mini GRU Cell (Forward Pass) | Implement a GRU cell forward pass with reset, update, and candidate gates for one sequence timestep. | https://www.tensortonic.com/problems/gru-cell-forward |
| Implement Huber Loss | Compute Huber loss with quadratic errors near zero and linear penalties beyond a configurable threshold. | https://www.tensortonic.com/problems/huber-loss |
| Implement Leaky ReLU (with α) | Apply Leaky ReLU element-wise with a configurable negative slope while retaining positive inputs. | https://www.tensortonic.com/problems/leaky-relu |
| Logistic Regression Training Loop | Train binary logistic regression in NumPy using sigmoid probabilities, gradient descent, and learned weight and bias parameters. | https://www.tensortonic.com/problems/logistic-regression-training |
| Matrix Inverse | Compute a square matrix inverse in NumPy while returning no result for invalid, non-square, or singular inputs. | https://www.tensortonic.com/problems/matrix-inverse |
| Normalize 3D Vectors | Normalize a 3D vector to unit length in NumPy while returning the required result for a zero vector. | https://www.tensortonic.com/problems/normalize-3d |
| RNN Step Forward (Tanh Cell) | Implement one vanilla RNN timestep with affine input and recurrent transforms followed by tanh activation. | https://www.tensortonic.com/problems/rnn-step-forward |
| Implement Sigmoid in NumPy | Implement a vectorized sigmoid activation in NumPy for scalars, lists, vectors, and matrices, including large positive and negative inputs. | https://www.tensortonic.com/problems/sigmoid-numpy |
| Compute 3D Vector Norm | Compute the Euclidean norm of a 3D vector from the square root of summed squared coordinates. | https://www.tensortonic.com/problems/vector-norm-3d |
| BatchNorm in ResNet | Implement ResNet batch normalization with channel statistics, learned scale and bias, and training or inference behavior. | https://www.tensortonic.com/research/resnet/resnet-batch-norm |
| Bottleneck Block | Build a ResNet bottleneck block using 1x1 channel reduction, 3x3 convolution, and 1x1 channel expansion. | https://www.tensortonic.com/research/resnet/resnet-bottleneck |
| Convolutional Block | Implement a ResNet convolutional block with a projected shortcut that matches changed spatial and channel dimensions. | https://www.tensortonic.com/research/resnet/resnet-conv-block |
| Full ResNet Assembly | Assemble a ResNet forward pass from the stem, residual stages, global average pooling, and the classification head. | https://www.tensortonic.com/research/resnet/resnet-full-network |
| Identity Block | Implement a ResNet identity block with a three-layer bottleneck branch, batch normalization, ReLU, and an unchanged skip path. | https://www.tensortonic.com/research/resnet/resnet-identity-block |
| Skip Connection Analysis | Analyze ResNet skip connections by combining residual and identity tensors and tracking gradient flow through the addition. | https://www.tensortonic.com/research/resnet/resnet-skip-connection |
| Scaled Dot-Product Attention | Implement scaled dot-product attention in PyTorch using query-key scores, softmax weights, and value aggregation. | https://www.tensortonic.com/research/transformer/transformers-attention |
| Embedding Layer | Create PyTorch token embeddings and scale each lookup by the square root of the Transformer model dimension. | https://www.tensortonic.com/research/transformer/transformers-embedding |
| Encoder Block | Assemble a Transformer encoder block with multi-head attention, residual paths, layer normalization, and a feed-forward network. | https://www.tensortonic.com/research/transformer/transformers-encoder-block |
| Feed-Forward Network | Implement the Transformer's position-wise feed-forward network with two linear projections and a ReLU activation. | https://www.tensortonic.com/research/transformer/transformers-feed-forward |
| Layer Normalization | Implement Transformer layer normalization in NumPy using per-token mean, variance, scale, and bias. | https://www.tensortonic.com/research/transformer/transformers-layer-normalization |
| Multi-Head Attention | Build NumPy multi-head attention with learned projections, per-head scaled attention, concatenation, and output projection. | https://www.tensortonic.com/research/transformer/transformers-multi-head-attention |
| Positional Encoding | Implement sinusoidal Transformer positional encodings in NumPy with alternating sine and cosine dimensions. | https://www.tensortonic.com/research/transformer/transformers-positional-encoding |
| Tokenization | Build a word-level Transformer tokenizer with fixed special-token IDs, sorted vocabulary entries, encoding, and decoding. | https://www.tensortonic.com/research/transformer/transformers-tokenization |
| Aggregation Functions | Implement Aggregation Functions, and return a dict mapping each function name to a dict of group label to aggregated value. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-aggregation-functions |
| Boolean Indexing | Filter pandas rows by a numeric column threshold and return the matching records with their original column order. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-boolean-indexing |
| Change Data Types | Create a DataFrame, convert the specified column to the target type, and return the dtypes before and after conversion. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-change-dtypes |
| Column Selection | Create a pandas DataFrame from dictionary data and extract one named column as an ordered list. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-column-selection |
| Cross Tabulation | Create a DataFrame and compute a cross-tabulation (frequency table) showing how often each combination of values co-occurs. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-cross-tabulation |
| Data Types Overview | Create a pandas DataFrame and report each column dtype together with counts for every unique dtype. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-data-types |
| Drop Duplicates | Create a DataFrame, remove duplicate rows, and return the cleaned result along with counts of rows before and after deduplication. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-drop-duplicates |
| GroupBy Basics | Create a DataFrame and compute the sum, mean, and count of the value column for each group. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-groupby-basics |
| Handle Missing Values | Create a pandas DataFrame, count missing entries per column, and replace every null with a supplied fill value. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-handle-missing |
| Head and Tail Operations | Create a pandas DataFrame and return the requested first and last rows as record-oriented dictionaries. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-head-tail |
| Inspect DataFrame Shape | Create a DataFrame and return its structural properties: row count, column count, column names, data types, and total number of values. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-inspect-shape |
| Loc vs iLoc | Create a DataFrame and use positional indexing to extract: the single element, the full row, and the full column. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-loc-iloc |
| Multi-Column Selection | Create a pandas DataFrame and select an ordered subset of named columns without changing row order. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-multi-column-selection |
| Multi-Level GroupBy | Create a DataFrame, group by all specified columns, apply the aggregation, and return the result as a flat table. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-multi-level-groupby |
| Pivot Tables | Build a pandas pivot table with selected index, columns, values, aggregation, and zero-filled missing combinations. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-pivot-tables |
| Create DataFrame from Dict | Create a pandas DataFrame from dictionary data and report its records, shape, and ordered column names. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-read-csv |
| Rename Columns | Rename selected pandas DataFrame columns from an old-to-new mapping and return the updated records. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-rename-columns |
| Replace Values | Create a DataFrame, replace all occurrences of the old value with the new value in the specified column, and count how many replacements were made. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-replace-values |
| Resetting Index | Set a pandas column as the index, then restore the default integer index while retaining the original values. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-resetting-index |
| Setting Index | Set a named pandas DataFrame column as the index and report the resulting records and index metadata. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-setting-index |
| Basic SELECT | Write a SQL SELECT query that aliases product names and calculates inventory value from unit price and stock quantity. | https://www.tensortonic.com/study-plans/sql-basics/sql/sql-basic-select |
| Conditional Aggregation | Summarize support tickets by department with conditional SQL counts for open, in-progress, and closed statuses. | https://www.tensortonic.com/study-plans/sql-basics/sql/sql-conditional-aggregation |
| COUNT, SUM, AVG | Aggregate sales by category with SQL COUNT, SUM, and AVG while handling NULL discounts and deterministic ordering. | https://www.tensortonic.com/study-plans/sql-basics/sql/sql-count-sum-avg |
| Cross Join | Generate every segment and metric combination with a SQL CROSS JOIN for a complete reporting grid. | https://www.tensortonic.com/study-plans/sql-basics/sql/sql-cross-join |
| DISTINCT Values | Return each customer and their distinct product count with SQL aggregation and deterministic sorting. | https://www.tensortonic.com/study-plans/sql-basics/sql/sql-distinct-values |
| GROUP BY | Group orders by customer in SQL to calculate total order count and spending, ordered by highest spend. | https://www.tensortonic.com/study-plans/sql-basics/sql/sql-group-by |
| HAVING Clause | Use SQL GROUP BY and HAVING to find customers with at least two orders and summarize their total spending. | https://www.tensortonic.com/study-plans/sql-basics/sql/sql-having-clause |
| INNER JOIN | Join employees to matching departments with SQL INNER JOIN and return employee name, salary, and department. | https://www.tensortonic.com/study-plans/sql-basics/sql/sql-inner-join |
| LEFT JOIN | Use SQL LEFT JOIN to include every customer and calculate total spending, returning zero for customers without orders. | https://www.tensortonic.com/study-plans/sql-basics/sql/sql-left-join |
| LIMIT and OFFSET | Use SQL ORDER BY, LIMIT, and OFFSET to return the second through fourth highest-revenue sales with tie-breaking. | https://www.tensortonic.com/study-plans/sql-basics/sql/sql-limit-offset |
| Multiple Joins | Join users, experiment assignments, and conversion events in SQL to report converted users, variants, and revenue. | https://www.tensortonic.com/study-plans/sql-basics/sql/sql-multiple-joins |
| Nested Aggregations | Use a SQL subquery or CTE to compute daily order totals, average daily revenue, and the busiest day. | https://www.tensortonic.com/study-plans/sql-basics/sql/sql-nested-aggregations |
| ORDER BY | Sort student exam results in SQL by descending score and ascending name for deterministic ties. | https://www.tensortonic.com/study-plans/sql-basics/sql/sql-order-by |
| RANK and DENSE_RANK | Rank ML models within each dataset using SQL RANK and DENSE_RANK over descending accuracy. | https://www.tensortonic.com/study-plans/sql-basics/sql/sql-rank-dense-rank |
| ROW_NUMBER | Assign deterministic per-segment activity ranks with SQL ROW_NUMBER ordered by engagement score and username. | https://www.tensortonic.com/study-plans/sql-basics/sql/sql-row-number |
| Self Join | Use a SQL self join to pair users with their referrers while labeling organic signups without a referral. | https://www.tensortonic.com/study-plans/sql-basics/sql/sql-self-join |
| WHERE Clauses | Filter employees by department and salary with SQL WHERE conditions, returning only qualifying names and salaries. | https://www.tensortonic.com/study-plans/sql-basics/sql/sql-where-clauses |
| Gmail Spam Filter |  | https://www.tensortonic.com/system-design/spam-filter |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/pritamsba2027)
<!-- tensortonic:end -->
