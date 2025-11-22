import streamlit as st
# Page configuration
st.set_page_config(
    page_title="An Explainable Multi-Task Learning Approach for Behavioral Biometrics using Social Media Data",
    page_icon="magnifying_glass",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    .main {background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%); color: #e0e0e0; font-family: 'Inter', sans-serif;}
    .stApp {background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);}
    .header {background: rgba(26, 26, 46, 0.8); padding: 20px; text-align: center; border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.3); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); margin-bottom: 20px;}
    .section {background: rgba(255,255,255,0.05); padding: 18px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.2); margin-bottom: 25px; border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(5px);}
    .dataset-info {background: rgba(0,123,255,0.1); border-left: 4px solid #007bff; padding: 12px; border-radius: 6px; color: #d1e7ff; font-size: 0.95em;}
    .algo-section {background: rgba(75,0,130,0.1); border-left: 4px solid #4B0082; padding: 15px; border-radius: 8px; margin-bottom: 20px; color: #e6d9ff;}
    .equation {background: rgba(0,0,0,0.2); padding: 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.1); margin: 10px 0;}
    .footer {text-align: center; margin-top: 40px; padding: 15px; background: rgba(26,26,46,0.8); color: #a0a0a0; border-radius: 12px; font-size: 0.85em;}
    .graph-caption {font-size: 0.92em; color: #b0d0ff; margin-top: 8px; line-height: 1.45;}
    h1, h2 {color: #ffffff;}
    h2 {color: #007bff; border-bottom: 1px solid rgba(0,123,255,0.3); padding-bottom: 6px;}
    h3 {color: #4B0082; font-size: 1.1em; margin-top: 20px;}
    .stImage > img {border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.4);}
    pre {background: rgba(0,0,0,0.3); padding: 15px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); overflow-x: auto;}
    code {color: #00ff00; font-family: 'Courier New', monospace;}
    </style>
""", unsafe_allow_html=True)
# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", ["Dashboard", "Algorithms"])
# Header
st.markdown('<div class="header"><h1>An Explainable Multi-Task Learning Approach for<br>Behavioral Biometrics using Social Media Data</h1></div>', unsafe_allow_html=True)
if page == "Dashboard":
    # Overview
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("### Overview")
        st.write("This dashboard reveals deep behavioral biometric patterns from large-scale Reddit education discussions (2010–2024). Temporal rhythms, engagement surges, power-law user activity, and social network structures highlight how users interact with educational content online.")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="dataset-info">
            <strong>Dataset Summary</strong><br>
            • Total Posts: 400,000+<br>
            • Unique Users: 3,464<br>
            • Topics Covered: 25<br>
            • Time Span: 2010–2024
        </div>
        """, unsafe_allow_html=True)
    # 1. Temporal Posting Patterns
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("## Temporal Posting Patterns")
    st.write("Clear circadian rhythms emerge: peak posting occurs between 5–7 PM, while engagement per post is highest around noon — suggesting users browse deeply during work/school breaks and post casually after hours.")
    cols = st.columns(3)
    with cols[0]:
        st.image("Posting_activity_by_hour_ofDay(All Users).png", use_container_width=True)
        st.markdown("<p class='graph-caption'><strong>Posting Activity by Hour</strong><br>Strong late-afternoon/evening peak (5–6 PM) across all users — typical post-work/post-school behavior.</p>", unsafe_allow_html=True)
    with cols[1]:
        st.image("usage_time_of_posting.png", use_container_width=True)
        st.markdown("<p class='graph-caption'><strong>Usage Time Distribution</strong><br>Heatmap confirms consistent daily cycles with weekend vs weekday variations visible.</p>", unsafe_allow_html=True)
    with cols[2]:
        st.image("Average_Engagement_perpost_byHour.png", use_container_width=True)
        st.markdown("<p class='graph-caption'><strong>Avg Engagement per Post by Hour</strong><br>Midday posts (11 AM–2 PM) receive significantly higher interaction — ideal time for impactful content.</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    # 2. Topic-Specific Posting Hours
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("## Topic-Specific Posting Hours")
    st.write("Different education topics show distinct temporal fingerprints — AI, ChatGPT, and EdTech peak sharply in evenings, while emotional intelligence and media literacy are more evenly distributed.")
    cols = st.columns(3)
    with cols[0]:
        st.image("posting_hours_for_topics_part_1.png", use_container_width=True)
        st.markdown("<p class='graph-caption'><strong>AI → Blockchain Topics</strong><br>AI-related discussions surge after 6 PM — users explore emerging tech in personal time.</p>", unsafe_allow_html=True)
    with cols[1]:
        st.image("posting_hours_for_topics_part_2.png", use_container_width=True)
        st.markdown("<p class='graph-caption'><strong>ChatGPT & EdTech</strong><br>ChatGPT shows extreme evening spike — reflective of after-class experimentation.</p>", unsafe_allow_html=True)
    with cols[2]:
        st.image("posting_hours_for_topics_part_3.png", use_container_width=True)
        st.markdown("<p class='graph-caption'><strong>Emotional Intelligence → Media Literacy</strong><br>More balanced distribution — these topics are discussed throughout the day.</p>", unsafe_allow_html=True)
    cols = st.columns(2)
    with cols[0]:
        st.image("posting_hours_for_topics_part_4.png", use_container_width=True)
        st.markdown("<p class='graph-caption'><strong>Microlearning → Remote Learning</strong><br>Remote learning peaks early evening — aligns with online course access patterns.</p>", unsafe_allow_html=True)
    with cols[1]:
        st.image("posting_hours_for_topics_part_5.png", use_container_width=True)
        st.markdown("<p class='graph-caption'><strong>Libraries → Teacher Development</strong><br>Professional topics show morning and evening bimodal activity — likely teachers planning lessons.</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    # 3. Engagement Biometrics
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("## Engagement Biometrics")
    st.write("Total engagement skyrocketed post-2020 due to pandemic-driven online shift and the 2023 AI boom. Remote Learning and Community-Based Learning dominate interaction volume.")
    cols = st.columns(3)
    with cols[0]:
        st.image("Engagement_over_time.png", use_container_width=True)
        st.markdown("<p class='graph-caption'><strong>Engagement Over Time (2010–2024)</strong><br>Massive spike after 2020 (pandemic) and explosive growth from 2023 (ChatGPT effect).</p>", unsafe_allow_html=True)
    with cols[1]:
        st.image("total_engagement_by_class(log_scale).png", use_container_width=True)
        st.markdown("<p class='graph-caption'><strong>Total Engagement by Class (Log Scale)</strong><br>Extreme inequality: top topics receive millions more interactions than niche ones.</p>", unsafe_allow_html=True)
    with cols[2]:
        st.image("total_engagement.png", use_container_width=True)
        st.markdown("<p class='graph-caption'><strong>Engagements by Topic</strong><br>Remote Learning and Community-Based Learning each exceed 119M total interactions.</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    # 4. User & Topic Distribution
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("## User & Topic Distribution")
    st.write("Classic power-law behavior: a tiny fraction of users generate most content. Sustainability Education leads in post volume, driven heavily by AutoModerator.")
    cols = st.columns(3)
    with cols[0]:
        st.image("PostsperUserDistribution(Log-Log Scaler)-Clear Long tail Behaviour.png", use_container_width=True)
        st.markdown("<p class='graph-caption'><strong>Posts per User (Log-Log)</strong><br>Perfect straight line = strong long-tail. Top 1% of users create majority of content.</p>", unsafe_allow_html=True)
    with cols[1]:
        st.image("Top15MostActiveUsers.png", use_container_width=True)
        st.markdown("<p class='graph-caption'><strong>Top 15 Most Active Users</strong><br>AutoModerator dominates, followed by dedicated community moderators and educators.</p>", unsafe_allow_html=True)
    with cols[2]:
        st.image("Number_Of_Post_per_Topics(25Classes).png", use_container_width=True)
        st.markdown("<p class='graph-caption'><strong>Posts per Topic (25 Classes)</strong><br>All the topics are with ~16K posts, showing strong community focus.</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    # 5. Network Forensics
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("## Network Forensics")
    st.write("Social and semantic networks reveal hidden influencers, resource sharing hubs, and cross-topic connections driving educational discourse.")
    cols = st.columns(3)
    with cols[0]:
        st.image("user-URL_Network.png", use_container_width=True)
        st.markdown("<p class='graph-caption'><strong>User-URL Affinity Network</strong><br>Central users act as knowledge bridges, consistently sharing high-value external resources.</p>", unsafe_allow_html=True)
    with cols[1]:
        st.image("hashtag_co-occurrence_network.png", use_container_width=True)
        st.markdown("<p class='graph-caption'><strong>Hashtag Co-occurrence</strong><br>Dense clusters around #AI, #EdTech, #RemoteLearning — showing thematic alignment.</p>", unsafe_allow_html=True)
    with cols[2]:
        st.image("Topic_mentioned_network.png", use_container_width=True)
        st.markdown("<p class='graph-caption'><strong>Topic Mention Network</strong><br>Reveals which topics are frequently discussed together (e.g., AI + Ethics, Gamification + Engagement).</p>", unsafe_allow_html=True)
    cols = st.columns(2)
    with cols[0]:
        st.image("User_Mention_Network.png", use_container_width=True)
        st.markdown("<p class='graph-caption'><strong>User Mention Network (Filtered)</strong><br>Core influencers and community leaders clearly visible at network center.</p>", unsafe_allow_html=True)
    with cols[1]:
        st.image("Group_engagement_network.png", use_container_width=True)
        st.markdown("<p class='graph-caption'><strong>Group Engagement Network</strong><br>Shows cross-subreddit knowledge flow and community overlap in education discourse.</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
elif page == "Algorithms":
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("## Algorithms & Pseudocode")
    st.markdown("""
    This section provides a detailed exposition of the core algorithms and pseudocode employed in the multi-task learning framework for behavioral biometrics. Each algorithm is described with its purpose, step-by-step breakdown, relevant mathematical formulations (where applicable), and integration notes. These components address key challenges such as class imbalance, task interference in multi-task learning, efficient training pipelines, weighted sampling, and model interpretability via LIME.
    """)
    # 1. ComputeEffectiveClassWeights
    st.markdown('<div class="algo-section">', unsafe_allow_html=True)
    st.markdown("### 1. Effective Class Weights Computation for Imbalanced Datasets")
    st.markdown("""
    **Purpose:** In multi-class classification tasks with severe imbalance (e.g., rare topics or infrequent users), standard loss functions bias toward majority classes. This algorithm computes *effective* class weights using a label-smoothing approach inspired by Cui et al. (2019), which models the effective number of samples per class to dynamically adjust weights, mitigating under-representation without oversampling. It is crucial for both topic (25 classes) and user (3,776 classes) prediction tasks, ensuring equitable learning across skewed distributions.
    **Key Equation:** The effective number of samples for a class $c$ with $n_c$ instances is given by:""")
    st.latex(r"N_{eff}(c) = \frac{1 - \beta^{n_c}}{1 - \beta}")
    st.markdown("""
    where $\\beta = 0.97$ is a hyperparameter controlling smoothing (closer to 1 yields finer granularity). The inverse, $w_c = 1 / N_{eff}(c)$, serves as the base weight. Optional log-scaling applies $\\log(1 + w_c)$ for numerical stability in extreme imbalances.
    **Step-by-Step Breakdown:**
    1. **Frequency Counting:** Tally occurrences of each class in the label array using a counter.
    2. **Weight Initialization:** Create an empty dictionary for weights.
    3. **Per-Class Computation:** For each class $c$ from 0 to $num\\_classes - 1$:
       - Retrieve count $n_c$ (default 0 if absent).
       - If $n_c > 0$, compute effective samples $N_{eff}(c)$ and weight $w_c = 1 / N_{eff}(c)$; else, set $w_c = 1.0$.
       - Apply log-scaling if enabled: $w_c \\leftarrow \\log(1 + w_c)$.
       - Assign to dictionary.
    4. **Normalization:** Scale all weights by the maximum value to bound them in [0, 1].
    5. **Output:** Return the normalized weights dictionary.
    **Integration & Usage (Eq. 11):** Weights are computed separately for topics and users, then combined for sample-level weighting:""")
    st.latex(r"\mathbf{w}_s = \alpha \cdot \mathbf{w}_{topic}(y_t) + \beta \cdot \mathbf{w}_{user}(y_u)")
    st.markdown("""
    where $\\alpha = 0.5$, $\\beta = 0.5$ (tunable), $y_t$ is topic label, and $y_u$ is user label. These are passed to the DataGenerator for weighted loss computation.
    **Pseudocode:**
    """)
    st.code("""
ALGORITHM: ComputeEffectiveClassWeights(labels, β=0.97, use_log_scaling=True, num_classes)
INPUT: labels (array of class indices), β (smoothing factor), use_log_scaling (bool), num_classes (int)
OUTPUT: class_weights (dict: class_id → normalized weight)
1. class_counter = count(labels) // count occurrences per class
2. class_weights = empty dict
3. FOR each c in 0 to num_classes-1:
      count_c = class_counter[c] or 0
      IF count_c > 0:
         effective_num = (1 - β**count_c) / (1 - β)
         weight = 1 / effective_num
      ELSE:
         weight = 1.0
      IF use_log_scaling:
         weight = log(1 + weight)
      class_weights[c] = weight
4. max_weight = max(class_weights.values())
   FOR each c in class_weights:
      class_weights[c] = class_weights[c] / max_weight // normalize
5. RETURN class_weights
// Usage for combined sample weights (Eq. 11)
topic_weights = ComputeEffectiveClassWeights(topic_labels, num_classes=25)
username_weights = ComputeEffectiveClassWeights(username_labels, num_classes=3776)
sample_weights = [α * topic_weights[label_t] + β * username_weights[label_u] for label_t, label_u in zip(topic_batch, username_batch)]
// where α=0.5, β=0.5, tunable
    """, language="python")
    st.markdown('</div>', unsafe_allow_html=True)
    # 2. OrthogonalizationLossLayer
    st.markdown('<div class="algo-section">', unsafe_allow_html=True)
    st.markdown("### 2. Orthogonalization Loss Layer for Task Decoupling")
    st.markdown("""
    **Purpose:** In multi-task learning (MTL), shared representations can lead to negative transfer, where one task (e.g., topic classification) interferes with another (e.g., user identification). This custom Keras layer enforces orthogonality between task-specific feature branches by penalizing their cosine similarity, promoting independent learning while retaining shared low-level features. The loss is added as an auxiliary term to the total objective.
    **Key Equation:** Cosine similarity between topic features $\\mathbf{f}_t$ and user features $\\mathbf{f}_u$ (both $\\mathbb{R}^{128}$) for a batch is:""")
    st.latex(r"\cos(\mathbf{f}_t, \mathbf{f}_u) = \frac{\mathbf{f}_t \cdot \mathbf{f}_u}{\|\mathbf{f}_t\| \cdot \|\mathbf{f}_u\|}")
    st.markdown("""
    The orthogonality loss is the mean absolute similarity:""")
    st.latex(r"\mathcal{L}_{ortho} = \frac{1}{B} \sum_{b=1}^B |\cos(\mathbf{f}_t^{(b)}, \mathbf{f}_u^{(b)})|")
    st.markdown("""
    where $B$ is batch size, weighted by $\\lambda = 0.1$ and added to the primary cross-entropy losses.
    **Step-by-Step Breakdown:**
    1. **Input Parsing:** Receive topic and user feature tensors from respective branches.
    2. **Dot Product:** Compute element-wise dot products along the feature dimension.
    3. **Normalization:** Calculate L2 norms for each sample, adding $\\epsilon = 10^{-8}$ for stability.
    4. **Cosine Similarity:** Divide dot products by normalized norms.
    5. **Loss Computation:** Take mean of absolute similarities across the batch.
    6. **Integration:** Scale by weight and add to the model's total loss.
    7. **Pass-Through:** Return unmodified inputs to downstream layers.
    **Integration:** Applied post-branching in the MTL model, before final classifiers. This ensures feature spaces remain decorrelated, improving generalization (e.g., reducing task confusion in behavioral biometrics).
    **Pseudocode:**
    """)
    st.code("""
CLASS: OrthogonalizationLossLayer(weight=0.1)
INPUT: topic_feats (tensor: B × 128), username_feats (tensor: B × 128)
OUTPUT: inputs (unchanged), with added loss
METHOD: call(inputs)
1. topic_feats, username_feats = inputs
2. dot_product = sum(topic_feats * username_feats, axis=-1) // element-wise dot
3. topic_norms = sqrt(sum(topic_feats**2, axis=-1)) + ε // ε=1e-8
4. username_norms = sqrt(sum(username_feats**2, axis=-1)) + ε
5. cosine_sim = dot_product / (topic_norms * username_norms)
6. ortho_loss = mean(abs(cosine_sim)) // Penalize correlation
7. self.add_loss(weight * ortho_loss) // Add to model total loss
8. RETURN inputs // pass-through
// Model Integration:
ortho_layer = OrthogonalizationLossLayer(weight=0.1)([topic_branch, username_branch])
topic_out = Dense(25, activation='softmax')(topic_branch)
username_out = Dense(3776, activation='softmax')(username_branch)
    """, language="python")
    st.markdown('</div>', unsafe_allow_html=True)
    # 3. MTLTrainingPipeline
    st.markdown('<div class="algo-section">', unsafe_allow_html=True)
    st.markdown("### 3. Multi-Task Learning Training Pipeline")
    st.markdown("""
    **Purpose:** This end-to-end pipeline orchestrates preprocessing, model construction, imbalance handling, and training for the dual-task MTL model (topic classification + user identification). It leverages BERT embeddings for textual features, hard parameter sharing up to branch divergence, and weighted losses to achieve high accuracy despite class imbalance and task heterogeneity.
    **Key Aspects:** No explicit new equations, but integrates prior components. Loss weights balance tasks: topic (0.8) vs. user (1.2), reflecting user ID's higher cardinality. Optimizer: Adam with LR=3e-5; callbacks prevent overfitting.
    **Step-by-Step Breakdown:**
    1. **Preprocessing:** Clean text (lowercase, remove URLs/stopwords, stem); tokenize via BERT (pad/truncate to 128 tokens); stratified split (80/10/10) preserving topic/user distributions.
    2. **Encoding & Weights:** One-hot encode labels; compute effective weights using Algorithm 1.
    3. **Data Generators:** Instantiate weighted generators (Algorithm 4) for train/val.
    4. **Model Architecture:** Input (128); BERT + BN/Dropout/Dense backbone; parallel branches (2048 → residual → 128) with GELU activations; apply orthogonality (Algorithm 2); softmax outputs. Compile with weighted CE loss.
    5. **Training:** Fit with early stopping (patience=10 on val_topic_acc) and LR reduction.
    6. **Output:** Trained Keras model.
    **Integration:** Serves as the core training script, enabling reproducible MTL for behavioral analysis.
    **Pseudocode:**
    """)
    st.code("""
ALGORITHM: MTLTrainingPipeline(X_text, y_topic, y_username)
INPUT: X_text (posts), y_topic (25-class), y_username (3776-class)
OUTPUT: Trained model
1. // Preprocessing (III-A-C)
   X_clean = lowercase(remove_noise(stem(X_text))) // URLs, stopwords
   X_tok = BERT_tokenizer(X_clean, pad=True, trunc=True, max_len=128)
   splits = stratified_split(X_tok, y_topic, y_username, train=0.8, val=0.1, test=0.1, seed=42)
2. // Encoding & Imbalance (III-D-E)
   y_topic_cat = one_hot(y_topic_train, 25); y_username_cat = one_hot(y_username_train, 3776)
   w_topic = ComputeEffectiveClassWeights(y_topic_train, num_classes=25)
   w_username = ComputeEffectiveClassWeights(y_username_train, num_classes=3776)
3. // Generators
   train_gen = DataGenerator(X_train, y_topic_cat_train, y_username_cat_train, w_topic, w_username, batch=128, α=0.5, β=0.5)
   val_gen = DataGenerator(X_val, ..., shuffle=False)
4. // Model (III-F, Fig. 2)
   inp = Input(shape=(128,))
   backbone = BERT(inp) → BN → Dropout(0.2) → Dense(256) → BN → GELU → Dropout(0.2) → Dense(128)
   topic_branch = Dense(2048) → BN → GELU → Dropout(0.25) → Residual → Dense(128) → BN → GELU
   username_branch = similar to topic_branch
   ortho = OrthogonalizationLossLayer(0.1)([topic_branch, username_branch])
   topic_out = Dense(25, softmax)(topic_branch)
   username_out = Dense(3776, softmax)(username_branch)
   model = Model(inp, [topic_out, username_out])
   model.compile(Adam(lr=3e-5), loss='categorical_crossentropy', loss_weights={'topic_out': 0.8, 'username_out': 1.2})
5. // Training
   callbacks = [EarlyStopping(monitor='val_topic_accuracy', patience=10), ReduceLROnPlateau(patience=10)]
   history = model.fit(train_gen, validation_data=val_gen, epochs=100, callbacks=callbacks)
6. RETURN model
    """, language="python")
    st.markdown('</div>', unsafe_allow_html=True)
    # 4. DataGenerator
    st.markdown('<div class="algo-section">', unsafe_allow_html=True)
    st.markdown("### 4. Weighted DataGenerator for Imbalanced MTL")
    st.markdown("""
    **Purpose:** Custom Keras sequence for efficient batching in imbalanced MTL, incorporating sample weights from Algorithm 1 to upweight minority classes during gradient computation. Supports shuffling and on-the-fly weighting, reducing memory footprint for large datasets (400K+ posts).
    **Key Equation:** Per-sample weight (from Eq. 11):""")
    st.latex(r"w_s^{(i)} = \alpha \cdot w_{topic}^{(y_t^i)} + \beta \cdot w_{user}^{(y_u^i)}")
    st.markdown("""
    Applied in weighted cross-entropy:""")
    st.latex(r"\mathcal{L} = -\sum w_s^{(i)} \cdot y^{(i)} \log(\hat{y}^{(i)})")
    st.markdown("""
    **Step-by-Step Breakdown:**
    1-4. **Batching:** Slice indices for current batch; extract X, y_topic, y_username.
    5-6. **Label Extraction:** Argmax one-hots to get integer labels.
    7-8. **Weight Lookup:** Map labels to precomputed weights.
    9. **Combination:** Linear fusion via $\\alpha$, $\\beta$ (note: $\\beta=0.2$ in class def, but tunable).
    10. **Yield:** Batch with outputs dict and weights for model.fit.
    **Integration:** Used in pipeline step 3; enables focal learning on rare users/topics.
    **Pseudocode:**
    """)
    st.code("""
CLASS: DataGenerator(batch_size=128, shuffle=True, α=0.5, β=0.5)
INPUT: X (tokens), y_topic_cat (one-hot), y_username_cat (one-hot), w_topic, w_username
OUTPUT: (X_b, {'topic_out': y_topic_b, 'username_out': y_username_b}, w_s)
INIT: indexes = range(len(X)); on_epoch_end: if shuffle, shuffle(indexes)
METHOD: __getitem__(idx):
   1. batch_idxs = indexes[idx*batch_size : (idx+1)*batch_size]
   2. X_b = X[batch_idxs]
   3. y_topic_b = y_topic_cat[batch_idxs]
   4. y_username_b = y_username_cat[batch_idxs]
   5. topic_lbls = argmax(y_topic_b, axis=-1)
   6. username_lbls = argmax(y_username_b, axis=-1)
   7. topic_wts = [w_topic[l] for l in topic_lbls]
   8. username_wts = [w_username[l] for l in username_lbls]
   9. w_s = α * topic_wts + β * username_wts // for weighted loss
  10. RETURN X_b, {'topic_out': y_topic_b, 'username_out': y_username_b}, w_s
    """, language="python")
    st.markdown('</div>', unsafe_allow_html=True)
    # 5. LIMEExplainability
    st.markdown('<div class="algo-section">', unsafe_allow_html=True)
    st.markdown("### 5. LIME-Based Explainability for MTL Interpretability")
    st.markdown("""
    **Purpose:** Post-hoc interpretability is vital for behavioral biometrics to uncover feature-task alignments. This algorithm uses Local Interpretable Model-agnostic Explanations (LIME; Ribeiro et al., 2016) to approximate task-specific predictions locally via interpretable surrogates (e.g., TF-IDF + linear model), revealing top influential words and overlaps between tasks.
    **Key Equations:** Feature importance variance across users/tasks (Eq. 6-7):""")
    st.latex(r"\Delta w_f = |w_f^{topic} - w_f^{user}| \quad (Eq. 6)")
    st.latex(r"\sigma^2_w = \frac{1}{F} \sum_{f=1}^F (\Delta w_f - \bar{\Delta w})^2 \quad (Eq. 7)")
    st.markdown("""
    where $w_f$ is weight for feature $f$, measuring stability/explainability.
    **Step-by-Step Breakdown:**
    1. **Topic Explainer:** Train LIME on training BERT features (TF-IDF mode); explain test instance via topic predictor, extract top-N words.
    2. **User Explainer:** Similarly for user predictor.
    3. **Overlap Analysis:** Compute set intersections for shared/unique features; visualize Venn (e.g., shared=14, unique=16 each).
    4. **Metrics:** Per-user word weight dict; compute variances (Eq. 7).
    5. **Output:** Explanations, sets; bar plots/Venn for visualization (Figs. 6-8).
    **Integration:** Applied post-training on test samples to audit model decisions, e.g., why a post is attributed to a user/topic.
    **Pseudocode:**
    """)
    st.code("""
ALGORITHM: LIMEExplainability(model, X_test, y_true_topic, y_true_username, num_features=10)
INPUT: model, X_test (tokenized), true labels
OUTPUT: explanations, shared/unique features
1. // Topic (Fig. 6)
   expl_topic = LimeTabularExplainer(X_train, mode='classification', num_features=5000) // TF-IDF
   exp_topic = expl_topic.explain_instance(X_test, model.predict_topic, num_features=num_features)
   top_topic = exp_topic.as_list() // [('word1', wt1), ...]
   // Bar plot: words vs. weights
2. // Username (Fig. 7)
   exp_user = expl_topic.explain_instance(X_test, model.predict_username, num_features=num_features) // Reuse or new
   top_user = exp_user.as_list() // [('word2', wt2), ...]
3. // Overlap (Fig. 8, Venn)
   shared = set(top_topic.keys()) ∩ set(top_user.keys()) // e.g., 14
   unique_topic = set(top_topic.keys()) - shared // 16
   unique_user = set(top_user.keys()) - shared // 16
   // Venn: shared=14, unique_t=16, unique_u=16
4. // Metrics (Eq. 6-7)
   delta_w = {f: abs(top_topic.get(f,0) - top_user.get(f,0)) for f in set(top_topic) ∪ set(top_user)}
   var_w = variance(list(delta_w.values())) // per-user/task
5. RETURN {exp_topic, exp_user}, {shared, unique_topic, unique_user}
    """, language="python")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
# Footer
st.markdown('<div class="footer"><p>© 2025 Behavioral Biometrics Lab • All visualizations on one scrollable page • Powered by Streamlit & xAI • Data: Reddit Education Communities (2010–2024)</p></div>', unsafe_allow_html=True)