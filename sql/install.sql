-- REMLight Database Installation Script
-- Minimal tables: ontology, resources, sessions, messages, kv_store
-- With embeddings, triggers, and rem functions

-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================
-- TABLES
-- ============================================

-- KV Store (regular table, not unlogged)
CREATE TABLE IF NOT EXISTS kv_store (
    entity_key VARCHAR(512) PRIMARY KEY,
    entity_type VARCHAR(128) NOT NULL,
    table_name VARCHAR(128),
    data JSONB NOT NULL DEFAULT '{}',
    user_id VARCHAR(256),
    tenant_id VARCHAR(256),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS kv_store_user_id_idx ON kv_store(user_id);
CREATE INDEX IF NOT EXISTS kv_store_entity_type_idx ON kv_store(entity_type);

-- Ontology (domain entities: people, projects, concepts)
CREATE TABLE IF NOT EXISTS ontologies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(512) NOT NULL,
    content TEXT,  -- Full markdown content
    description TEXT,  -- Short description for search
    category VARCHAR(256),
    entity_type VARCHAR(128),
    uri VARCHAR(1024),  -- Source file URI
    properties JSONB DEFAULT '{}',
    graph_edges JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    embedding VECTOR(1536),
    user_id VARCHAR(256),
    tenant_id VARCHAR(256),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);
CREATE UNIQUE INDEX IF NOT EXISTS ontologies_name_unique ON ontologies(name);
CREATE INDEX IF NOT EXISTS ontologies_user_id_idx ON ontologies(user_id);
CREATE INDEX IF NOT EXISTS ontologies_category_idx ON ontologies(category);
CREATE INDEX IF NOT EXISTS ontologies_embedding_idx ON ontologies USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS ontologies_graph_edges_idx ON ontologies USING GIN (graph_edges);

-- Resources (documents, content chunks)
CREATE TABLE IF NOT EXISTS resources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(512),
    uri VARCHAR(1024),
    ordinal INTEGER DEFAULT 0,
    content TEXT,
    category VARCHAR(256),
    related_entities JSONB DEFAULT '[]',
    graph_edges JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    embedding VECTOR(1536),
    user_id VARCHAR(256),
    tenant_id VARCHAR(256),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE,
    CONSTRAINT resource_unique_uri_ordinal UNIQUE (user_id, uri, ordinal)
);
CREATE UNIQUE INDEX IF NOT EXISTS resources_name_unique ON resources(name) WHERE name IS NOT NULL;
CREATE INDEX IF NOT EXISTS resources_user_id_idx ON resources(user_id);
CREATE INDEX IF NOT EXISTS resources_category_idx ON resources(category);
CREATE INDEX IF NOT EXISTS resources_embedding_idx ON resources USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS resources_graph_edges_idx ON resources USING GIN (graph_edges);

-- Users (user profiles)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(512),
    email VARCHAR(256) UNIQUE,
    summary TEXT,  -- AI-generated profile summary
    interests TEXT[] DEFAULT '{}',
    preferred_topics TEXT[] DEFAULT '{}',
    activity_level VARCHAR(64),
    metadata JSONB DEFAULT '{}',
    user_id VARCHAR(256),
    tenant_id VARCHAR(256),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);
CREATE INDEX IF NOT EXISTS users_email_idx ON users(email);
CREATE INDEX IF NOT EXISTS users_user_id_idx ON users(user_id);

-- Sessions (conversation sessions)
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(512),
    description TEXT,
    agent_name VARCHAR(256),
    status VARCHAR(64) DEFAULT 'active',
    graph_edges JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    user_id VARCHAR(256),
    tenant_id VARCHAR(256),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);
CREATE INDEX IF NOT EXISTS sessions_user_id_idx ON sessions(user_id);
CREATE INDEX IF NOT EXISTS sessions_status_idx ON sessions(status);

-- Messages (chat messages in sessions)
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    role VARCHAR(64) NOT NULL, -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    tool_calls JSONB DEFAULT '[]',
    graph_edges JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    embedding VECTOR(1536),
    trace_id VARCHAR(256),
    span_id VARCHAR(256),
    user_id VARCHAR(256),
    tenant_id VARCHAR(256),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);
CREATE INDEX IF NOT EXISTS messages_session_id_idx ON messages(session_id);
CREATE INDEX IF NOT EXISTS messages_user_id_idx ON messages(user_id);
CREATE INDEX IF NOT EXISTS messages_role_idx ON messages(role);
CREATE INDEX IF NOT EXISTS messages_embedding_idx ON messages USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Scenarios (labeled sessions for search and replay)
CREATE TABLE IF NOT EXISTS scenarios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(512),
    description TEXT,
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    agent_name VARCHAR(256),
    status VARCHAR(64) DEFAULT 'active',
    graph_edges JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    embedding VECTOR(1536),
    user_id VARCHAR(256),
    tenant_id VARCHAR(256),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);
CREATE INDEX IF NOT EXISTS scenarios_user_id_idx ON scenarios(user_id);
CREATE INDEX IF NOT EXISTS scenarios_session_id_idx ON scenarios(session_id);
CREATE INDEX IF NOT EXISTS scenarios_status_idx ON scenarios(status);
CREATE INDEX IF NOT EXISTS scenarios_tags_idx ON scenarios USING GIN (tags);
CREATE INDEX IF NOT EXISTS scenarios_created_at_idx ON scenarios(created_at);
CREATE INDEX IF NOT EXISTS scenarios_embedding_idx ON scenarios USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS scenarios_name_trgm_idx ON scenarios USING GIN (name gin_trgm_ops);

-- Agents (stored agent schemas from database)
CREATE TABLE IF NOT EXISTS agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(512) NOT NULL UNIQUE,  -- Unique constraint for ON CONFLICT
    description TEXT,  -- Used for semantic search (first line = title)
    content TEXT NOT NULL,  -- Full YAML content
    version VARCHAR(64) DEFAULT '1.0.0',
    enabled BOOLEAN DEFAULT TRUE,
    registry_uri VARCHAR(2048),  -- Registry source (NULL = "local")
    icon VARCHAR(512),           -- Icon URL or emoji
    graph_edges JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    embedding VECTOR(1536),
    user_id VARCHAR(256),
    tenant_id VARCHAR(256),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);
CREATE UNIQUE INDEX IF NOT EXISTS agents_name_unique ON agents(name) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS agents_user_id_idx ON agents(user_id);
CREATE INDEX IF NOT EXISTS agents_enabled_idx ON agents(enabled);
CREATE INDEX IF NOT EXISTS agents_tags_idx ON agents USING GIN (tags);
CREATE INDEX IF NOT EXISTS agents_embedding_idx ON agents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS agents_name_trgm_idx ON agents USING GIN (name gin_trgm_ops);

-- Files (uploaded/processed files with parsed output)
-- Uses URI hash as deterministic ID for upsert-by-URI pattern
CREATE TABLE IF NOT EXISTS files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(512) NOT NULL,  -- Original filename
    uri VARCHAR(2048) NOT NULL,  -- Source URI (s3://, file://, https://)
    uri_hash VARCHAR(64) NOT NULL,  -- SHA256 hash of URI for deduplication
    content TEXT,  -- Extracted text content
    mime_type VARCHAR(256),  -- MIME type (application/pdf, text/markdown, etc.)
    size_bytes BIGINT,  -- File size in bytes
    processing_status VARCHAR(64) DEFAULT 'pending',  -- pending, processing, completed, failed
    parsed_output JSONB DEFAULT '{}',  -- Rich parsing result (text, tables, images, metadata)
    graph_edges JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    user_id VARCHAR(256),
    tenant_id VARCHAR(256),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);
-- Note: Using regular UNIQUE constraint (not partial index) to support ON CONFLICT upsert
ALTER TABLE files ADD CONSTRAINT files_uri_hash_unique UNIQUE (uri_hash);
CREATE INDEX IF NOT EXISTS files_user_id_idx ON files(user_id);
CREATE INDEX IF NOT EXISTS files_processing_status_idx ON files(processing_status);
CREATE INDEX IF NOT EXISTS files_mime_type_idx ON files(mime_type);
CREATE INDEX IF NOT EXISTS files_name_trgm_idx ON files USING GIN (name gin_trgm_ops);

-- Servers (MCP tool server configurations)
CREATE TABLE IF NOT EXISTS servers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(512) NOT NULL,
    description TEXT,

    -- Server configuration
    server_type VARCHAR(64) DEFAULT 'mcp',  -- mcp (local), rest, stdio
    endpoint VARCHAR(2048),                   -- URL or command for remote/stdio
    config JSONB DEFAULT '{}',                -- Server-specific configuration
    enabled BOOLEAN DEFAULT TRUE,

    -- Federation support (future)
    registry_uri VARCHAR(2048),               -- Parent registry URI (nullable)

    -- Display
    icon VARCHAR(512),                        -- Icon URL or emoji

    -- System fields
    graph_edges JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    embedding VECTOR(1536),
    user_id VARCHAR(256),
    tenant_id VARCHAR(256),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);
-- Note: No unique constraint on name - using deterministic IDs from endpoint/name hash
-- This allows idempotent upserts by ID without conflict issues
CREATE INDEX IF NOT EXISTS servers_name_idx ON servers(name);
CREATE INDEX IF NOT EXISTS servers_user_id_idx ON servers(user_id);
CREATE INDEX IF NOT EXISTS servers_enabled_idx ON servers(enabled);
CREATE INDEX IF NOT EXISTS servers_server_type_idx ON servers(server_type);
CREATE INDEX IF NOT EXISTS servers_tags_idx ON servers USING GIN (tags);
CREATE INDEX IF NOT EXISTS servers_embedding_idx ON servers USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS servers_name_trgm_idx ON servers USING GIN (name gin_trgm_ops);

-- Tools (registered tool definitions)
CREATE TABLE IF NOT EXISTS tools (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(512) NOT NULL,
    description TEXT,

    -- Tool configuration
    server_id UUID REFERENCES servers(id) ON DELETE CASCADE,
    input_schema JSONB DEFAULT '{}',          -- JSON Schema for parameters
    enabled BOOLEAN DEFAULT TRUE,

    -- Display
    icon VARCHAR(512),

    -- System fields
    graph_edges JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    embedding VECTOR(1536),
    user_id VARCHAR(256),
    tenant_id VARCHAR(256),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);
CREATE UNIQUE INDEX IF NOT EXISTS tools_server_name_unique ON tools(server_id, name) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS tools_user_id_idx ON tools(user_id);
CREATE INDEX IF NOT EXISTS tools_server_id_idx ON tools(server_id);
CREATE INDEX IF NOT EXISTS tools_enabled_idx ON tools(enabled);
CREATE INDEX IF NOT EXISTS tools_tags_idx ON tools USING GIN (tags);
CREATE INDEX IF NOT EXISTS tools_embedding_idx ON tools USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS tools_name_trgm_idx ON tools USING GIN (name gin_trgm_ops);

-- Feedback (user feedback on agent responses)
-- Decoupled from Scenarios: Feedback is end-user ratings, Scenarios are admin test cases
CREATE TABLE IF NOT EXISTS feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    message_id UUID REFERENCES messages(id) ON DELETE SET NULL,
    trace_id VARCHAR(256),  -- Phoenix/OTEL trace ID
    span_id VARCHAR(256),   -- Phoenix/OTEL span ID
    name VARCHAR(256) DEFAULT 'user_feedback',  -- Annotation type
    score FLOAT,            -- Numeric rating (0.0 to 1.0)
    label VARCHAR(128),     -- Categorical label (thumbs_up, thumbs_down, helpful, etc.)
    comment TEXT,           -- Free text feedback
    source VARCHAR(64) DEFAULT 'user',  -- user, evaluator, automated
    graph_edges JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    user_id VARCHAR(256),
    tenant_id VARCHAR(256),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);
CREATE INDEX IF NOT EXISTS feedback_session_id_idx ON feedback(session_id);
CREATE INDEX IF NOT EXISTS feedback_message_id_idx ON feedback(message_id);
CREATE INDEX IF NOT EXISTS feedback_trace_id_idx ON feedback(trace_id);
CREATE INDEX IF NOT EXISTS feedback_user_id_idx ON feedback(user_id);
CREATE INDEX IF NOT EXISTS feedback_label_idx ON feedback(label);
CREATE INDEX IF NOT EXISTS feedback_source_idx ON feedback(source);
CREATE INDEX IF NOT EXISTS feedback_created_at_idx ON feedback(created_at);

-- Collections (groups of sessions for batch evaluation)
CREATE TABLE IF NOT EXISTS collections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(512) NOT NULL,
    description TEXT,
    session_count INTEGER DEFAULT 0,  -- Cached count, updated via triggers
    status VARCHAR(64) DEFAULT 'active',  -- active, archived, running, completed
    query_filter JSONB,  -- Saved query for auto-population
    graph_edges JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    embedding VECTOR(1536),  -- For semantic search on description
    user_id VARCHAR(256),
    tenant_id VARCHAR(256),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);
CREATE UNIQUE INDEX IF NOT EXISTS collections_name_unique ON collections(name) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS collections_user_id_idx ON collections(user_id);
CREATE INDEX IF NOT EXISTS collections_status_idx ON collections(status);
CREATE INDEX IF NOT EXISTS collections_tags_idx ON collections USING GIN (tags);
CREATE INDEX IF NOT EXISTS collections_embedding_idx ON collections USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS collections_name_trgm_idx ON collections USING GIN (name gin_trgm_ops);

-- Collection Sessions (junction table linking sessions to collections)
CREATE TABLE IF NOT EXISTS collection_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    ordinal INTEGER DEFAULT 0,  -- Ordering within collection
    notes TEXT,  -- Why session was included
    graph_edges JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    user_id VARCHAR(256),
    tenant_id VARCHAR(256),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE,
    CONSTRAINT collection_session_unique UNIQUE (collection_id, session_id)
);
CREATE INDEX IF NOT EXISTS collection_sessions_collection_id_idx ON collection_sessions(collection_id);
CREATE INDEX IF NOT EXISTS collection_sessions_session_id_idx ON collection_sessions(session_id);

-- Trigger to update collection session_count
CREATE OR REPLACE FUNCTION update_collection_session_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE collections SET session_count = session_count + 1, updated_at = NOW()
        WHERE id = NEW.collection_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE collections SET session_count = session_count - 1, updated_at = NOW()
        WHERE id = OLD.collection_id;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS collection_sessions_count_trigger ON collection_sessions;
CREATE TRIGGER collection_sessions_count_trigger
AFTER INSERT OR DELETE ON collection_sessions
FOR EACH ROW EXECUTE FUNCTION update_collection_session_count();

-- Agent Time Machine (version history for agents)
-- Note: No FK constraint on agent_id because we need to preserve history after agent deletion
CREATE TABLE IF NOT EXISTS agent_timemachine (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL,  -- Reference to agents.id (no FK - preserves history after deletion)
    agent_name VARCHAR(512) NOT NULL,
    content TEXT NOT NULL,  -- Full YAML content at this version
    version VARCHAR(64),
    content_hash VARCHAR(64) NOT NULL,  -- SHA256 hash for change detection
    change_type VARCHAR(64) NOT NULL,  -- 'created', 'updated', 'deleted'
    metadata JSONB DEFAULT '{}',
    user_id VARCHAR(256),
    tenant_id VARCHAR(256),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS agent_timemachine_agent_id_idx ON agent_timemachine(agent_id);
CREATE INDEX IF NOT EXISTS agent_timemachine_agent_name_idx ON agent_timemachine(agent_name);
CREATE INDEX IF NOT EXISTS agent_timemachine_created_at_idx ON agent_timemachine(created_at DESC);

-- ============================================
-- TRIGGERS: Auto-update KV Store
-- ============================================

-- Function to update kv_store from any table
CREATE OR REPLACE FUNCTION update_kv_store()
RETURNS TRIGGER AS $$
DECLARE
    entity_key_val VARCHAR(512);
BEGIN
    -- Generate entity key from name or id
    IF NEW.name IS NOT NULL AND NEW.name != '' THEN
        entity_key_val := lower(regexp_replace(NEW.name, '[^a-zA-Z0-9]+', '-', 'g'));
    ELSE
        entity_key_val := NEW.id::text;
    END IF;

    -- Upsert into kv_store
    INSERT INTO kv_store (entity_key, entity_type, table_name, data, user_id, tenant_id, updated_at)
    VALUES (
        entity_key_val,
        TG_TABLE_NAME,
        TG_TABLE_NAME,
        to_jsonb(NEW),
        NEW.user_id,
        NEW.tenant_id,
        NOW()
    )
    ON CONFLICT (entity_key) DO UPDATE SET
        data = to_jsonb(NEW),
        updated_at = NOW();

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for ontologies
DROP TRIGGER IF EXISTS ontology_kv_trigger ON ontologies;
CREATE TRIGGER ontology_kv_trigger
AFTER INSERT OR UPDATE ON ontologies
FOR EACH ROW EXECUTE FUNCTION update_kv_store();

-- Trigger for resources
DROP TRIGGER IF EXISTS resource_kv_trigger ON resources;
CREATE TRIGGER resource_kv_trigger
AFTER INSERT OR UPDATE ON resources
FOR EACH ROW EXECUTE FUNCTION update_kv_store();

-- Trigger for scenarios
DROP TRIGGER IF EXISTS scenario_kv_trigger ON scenarios;
CREATE TRIGGER scenario_kv_trigger
AFTER INSERT OR UPDATE ON scenarios
FOR EACH ROW EXECUTE FUNCTION update_kv_store();

-- Trigger for collections
DROP TRIGGER IF EXISTS collection_kv_trigger ON collections;
CREATE TRIGGER collection_kv_trigger
AFTER INSERT OR UPDATE ON collections
FOR EACH ROW EXECUTE FUNCTION update_kv_store();

-- Trigger for agents
DROP TRIGGER IF EXISTS agent_kv_trigger ON agents;
CREATE TRIGGER agent_kv_trigger
AFTER INSERT OR UPDATE ON agents
FOR EACH ROW EXECUTE FUNCTION update_kv_store();

-- Trigger for servers
DROP TRIGGER IF EXISTS server_kv_trigger ON servers;
CREATE TRIGGER server_kv_trigger
AFTER INSERT OR UPDATE ON servers
FOR EACH ROW EXECUTE FUNCTION update_kv_store();

-- Trigger for tools
DROP TRIGGER IF EXISTS tool_kv_trigger ON tools;
CREATE TRIGGER tool_kv_trigger
AFTER INSERT OR UPDATE ON tools
FOR EACH ROW EXECUTE FUNCTION update_kv_store();

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply updated_at trigger to all tables
DROP TRIGGER IF EXISTS ontologies_updated_at ON ontologies;
CREATE TRIGGER ontologies_updated_at BEFORE UPDATE ON ontologies
FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS resources_updated_at ON resources;
CREATE TRIGGER resources_updated_at BEFORE UPDATE ON resources
FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS sessions_updated_at ON sessions;
CREATE TRIGGER sessions_updated_at BEFORE UPDATE ON sessions
FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS messages_updated_at ON messages;
CREATE TRIGGER messages_updated_at BEFORE UPDATE ON messages
FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS scenarios_updated_at ON scenarios;
CREATE TRIGGER scenarios_updated_at BEFORE UPDATE ON scenarios
FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS agents_updated_at ON agents;
CREATE TRIGGER agents_updated_at BEFORE UPDATE ON agents
FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS kv_store_updated_at ON kv_store;
CREATE TRIGGER kv_store_updated_at BEFORE UPDATE ON kv_store
FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS files_updated_at ON files;
CREATE TRIGGER files_updated_at BEFORE UPDATE ON files
FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS servers_updated_at ON servers;
CREATE TRIGGER servers_updated_at BEFORE UPDATE ON servers
FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS tools_updated_at ON tools;
CREATE TRIGGER tools_updated_at BEFORE UPDATE ON tools
FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS feedback_updated_at ON feedback;
CREATE TRIGGER feedback_updated_at BEFORE UPDATE ON feedback
FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS collections_updated_at ON collections;
CREATE TRIGGER collections_updated_at BEFORE UPDATE ON collections
FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS collection_sessions_updated_at ON collection_sessions;
CREATE TRIGGER collection_sessions_updated_at BEFORE UPDATE ON collection_sessions
FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================
-- AGENT TIME MACHINE TRIGGER
-- ============================================
-- Records version history when agents are created, updated, or deleted
-- Only creates a new entry if the content has actually changed (via hash)

CREATE OR REPLACE FUNCTION record_agent_timemachine()
RETURNS TRIGGER AS $$
DECLARE
    new_hash VARCHAR(64);
    old_hash VARCHAR(64);
BEGIN
    -- Calculate hash of new content
    IF TG_OP = 'DELETE' THEN
        -- Record deletion
        INSERT INTO agent_timemachine (
            agent_id, agent_name, content, version, content_hash, change_type,
            metadata, user_id, tenant_id
        ) VALUES (
            OLD.id, OLD.name, OLD.content, OLD.version,
            encode(sha256(OLD.content::bytea), 'hex'),
            'deleted', OLD.metadata, OLD.user_id, OLD.tenant_id
        );
        RETURN OLD;
    END IF;

    new_hash := encode(sha256(NEW.content::bytea), 'hex');

    IF TG_OP = 'INSERT' THEN
        -- Record creation
        INSERT INTO agent_timemachine (
            agent_id, agent_name, content, version, content_hash, change_type,
            metadata, user_id, tenant_id
        ) VALUES (
            NEW.id, NEW.name, NEW.content, NEW.version, new_hash,
            'created', NEW.metadata, NEW.user_id, NEW.tenant_id
        );
    ELSIF TG_OP = 'UPDATE' THEN
        -- Only record if content actually changed
        old_hash := encode(sha256(OLD.content::bytea), 'hex');
        IF new_hash != old_hash THEN
            INSERT INTO agent_timemachine (
                agent_id, agent_name, content, version, content_hash, change_type,
                metadata, user_id, tenant_id
            ) VALUES (
                NEW.id, NEW.name, NEW.content, NEW.version, new_hash,
                'updated', NEW.metadata, NEW.user_id, NEW.tenant_id
            );
        END IF;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS agent_timemachine_trigger ON agents;
CREATE TRIGGER agent_timemachine_trigger
AFTER INSERT OR UPDATE OR DELETE ON agents
FOR EACH ROW EXECUTE FUNCTION record_agent_timemachine();

-- ============================================
-- REM FUNCTIONS
-- ============================================

-- rem_lookup: O(1) KV store lookup
CREATE OR REPLACE FUNCTION rem_lookup(
    p_key VARCHAR(512),
    p_user_id VARCHAR(256) DEFAULT NULL
)
RETURNS JSONB AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT data - 'embedding' INTO result
    FROM kv_store
    WHERE entity_key = p_key
      AND (p_user_id IS NULL OR user_id = p_user_id OR user_id IS NULL);

    RETURN COALESCE(result, '{}'::jsonb);
END;
$$ LANGUAGE plpgsql;

-- rem_search: Semantic vector search
CREATE OR REPLACE FUNCTION rem_search(
    p_query_embedding VECTOR(1536),
    p_table_name VARCHAR(128),
    p_limit INTEGER DEFAULT 10,
    p_min_similarity FLOAT DEFAULT 0.3,
    p_user_id VARCHAR(256) DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    name VARCHAR(512),
    content TEXT,
    similarity FLOAT,
    data JSONB
) AS $$
BEGIN
    IF p_table_name = 'ontologies' THEN
        RETURN QUERY
        SELECT o.id, o.name, COALESCE(o.description, o.content) as content,
               1 - (o.embedding <=> p_query_embedding) as similarity,
               to_jsonb(o) - 'embedding' as data
        FROM ontologies o
        WHERE o.embedding IS NOT NULL
          AND o.deleted_at IS NULL
          AND (p_user_id IS NULL OR o.user_id = p_user_id OR o.user_id IS NULL)
          AND 1 - (o.embedding <=> p_query_embedding) >= p_min_similarity
        ORDER BY o.embedding <=> p_query_embedding
        LIMIT p_limit;
    ELSIF p_table_name = 'resources' THEN
        RETURN QUERY
        SELECT r.id, r.name, r.content,
               1 - (r.embedding <=> p_query_embedding) as similarity,
               to_jsonb(r) - 'embedding' as data
        FROM resources r
        WHERE r.embedding IS NOT NULL
          AND r.deleted_at IS NULL
          AND (p_user_id IS NULL OR r.user_id = p_user_id OR r.user_id IS NULL)
          AND 1 - (r.embedding <=> p_query_embedding) >= p_min_similarity
        ORDER BY r.embedding <=> p_query_embedding
        LIMIT p_limit;
    ELSIF p_table_name = 'messages' THEN
        RETURN QUERY
        SELECT m.id, NULL::VARCHAR(512) as name, m.content,
               1 - (m.embedding <=> p_query_embedding) as similarity,
               to_jsonb(m) - 'embedding' as data
        FROM messages m
        WHERE m.embedding IS NOT NULL
          AND m.deleted_at IS NULL
          AND (p_user_id IS NULL OR m.user_id = p_user_id OR m.user_id IS NULL)
          AND 1 - (m.embedding <=> p_query_embedding) >= p_min_similarity
        ORDER BY m.embedding <=> p_query_embedding
        LIMIT p_limit;
    ELSIF p_table_name = 'scenarios' THEN
        RETURN QUERY
        SELECT s.id, s.name, s.description as content,
               1 - (s.embedding <=> p_query_embedding) as similarity,
               to_jsonb(s) - 'embedding' as data
        FROM scenarios s
        WHERE s.embedding IS NOT NULL
          AND s.deleted_at IS NULL
          AND (p_user_id IS NULL OR s.user_id = p_user_id OR s.user_id IS NULL)
          AND 1 - (s.embedding <=> p_query_embedding) >= p_min_similarity
        ORDER BY s.embedding <=> p_query_embedding
        LIMIT p_limit;
    ELSIF p_table_name = 'collections' THEN
        RETURN QUERY
        SELECT c.id, c.name, c.description as content,
               1 - (c.embedding <=> p_query_embedding) as similarity,
               to_jsonb(c) - 'embedding' as data
        FROM collections c
        WHERE c.embedding IS NOT NULL
          AND c.deleted_at IS NULL
          AND (p_user_id IS NULL OR c.user_id = p_user_id OR c.user_id IS NULL)
          AND 1 - (c.embedding <=> p_query_embedding) >= p_min_similarity
        ORDER BY c.embedding <=> p_query_embedding
        LIMIT p_limit;
    ELSIF p_table_name = 'agents' THEN
        RETURN QUERY
        SELECT a.id, a.name, a.description as content,
               1 - (a.embedding <=> p_query_embedding) as similarity,
               to_jsonb(a) - 'embedding' as data
        FROM agents a
        WHERE a.embedding IS NOT NULL
          AND a.deleted_at IS NULL
          AND a.enabled = TRUE
          AND (p_user_id IS NULL OR a.user_id = p_user_id OR a.user_id IS NULL)
          AND 1 - (a.embedding <=> p_query_embedding) >= p_min_similarity
        ORDER BY a.embedding <=> p_query_embedding
        LIMIT p_limit;
    ELSIF p_table_name = 'servers' THEN
        RETURN QUERY
        SELECT s.id, s.name, s.description as content,
               1 - (s.embedding <=> p_query_embedding) as similarity,
               to_jsonb(s) - 'embedding' as data
        FROM servers s
        WHERE s.embedding IS NOT NULL
          AND s.deleted_at IS NULL
          AND s.enabled = TRUE
          AND (p_user_id IS NULL OR s.user_id = p_user_id OR s.user_id IS NULL)
          AND 1 - (s.embedding <=> p_query_embedding) >= p_min_similarity
        ORDER BY s.embedding <=> p_query_embedding
        LIMIT p_limit;
    ELSIF p_table_name = 'tools' THEN
        RETURN QUERY
        SELECT t.id, t.name, t.description as content,
               1 - (t.embedding <=> p_query_embedding) as similarity,
               to_jsonb(t) - 'embedding' as data
        FROM tools t
        WHERE t.embedding IS NOT NULL
          AND t.deleted_at IS NULL
          AND t.enabled = TRUE
          AND (p_user_id IS NULL OR t.user_id = p_user_id OR t.user_id IS NULL)
          AND 1 - (t.embedding <=> p_query_embedding) >= p_min_similarity
        ORDER BY t.embedding <=> p_query_embedding
        LIMIT p_limit;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- rem_fuzzy: Fuzzy text search using trigrams
CREATE OR REPLACE FUNCTION rem_fuzzy(
    p_query_text TEXT,
    p_user_id VARCHAR(256) DEFAULT NULL,
    p_threshold FLOAT DEFAULT 0.3,
    p_limit INTEGER DEFAULT 20
)
RETURNS TABLE (
    entity_key VARCHAR(512),
    entity_type VARCHAR(128),
    similarity DOUBLE PRECISION,
    data JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT k.entity_key, k.entity_type,
           similarity(k.entity_key, p_query_text)::DOUBLE PRECISION as sim,
           k.data - 'embedding'
    FROM kv_store k
    WHERE similarity(k.entity_key, p_query_text) >= p_threshold
      AND (p_user_id IS NULL OR k.user_id = p_user_id OR k.user_id IS NULL)
    ORDER BY sim DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- rem_traverse: Graph traversal
CREATE OR REPLACE FUNCTION rem_traverse(
    p_entity_key VARCHAR(512),
    p_edge_types TEXT[] DEFAULT NULL,
    p_max_depth INTEGER DEFAULT 1,
    p_user_id VARCHAR(256) DEFAULT NULL
)
RETURNS TABLE (
    entity_key VARCHAR(512),
    entity_type VARCHAR(128),
    depth INTEGER,
    path TEXT[],
    data JSONB
) AS $$
WITH RECURSIVE traverse AS (
    -- Base case: start from the entity
    SELECT
        k.entity_key,
        k.entity_type,
        0 as depth,
        ARRAY[k.entity_key::TEXT] as path,
        k.data - 'embedding' as data
    FROM kv_store k
    WHERE k.entity_key = p_entity_key
      AND (p_user_id IS NULL OR k.user_id = p_user_id OR k.user_id IS NULL)

    UNION ALL

    -- Recursive case: follow graph_edges
    SELECT
        k.entity_key,
        k.entity_type,
        t.depth + 1,
        t.path || k.entity_key::TEXT,
        k.data - 'embedding' as data
    FROM traverse t
    CROSS JOIN LATERAL jsonb_array_elements(
        (SELECT data FROM kv_store WHERE entity_key = t.entity_key)->'graph_edges'
    ) as edge
    JOIN kv_store k ON k.entity_key = edge->>'target'
    WHERE t.depth < p_max_depth
      AND NOT k.entity_key::TEXT = ANY(t.path) -- Prevent cycles
      AND (p_edge_types IS NULL OR (edge->>'type') = ANY(p_edge_types))
      AND (p_user_id IS NULL OR k.user_id = p_user_id OR k.user_id IS NULL)
)
SELECT * FROM traverse;
$$ LANGUAGE sql;

-- ============================================
-- EMBEDDING QUEUE (for async embedding generation)
-- ============================================

CREATE TABLE IF NOT EXISTS embedding_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    table_name VARCHAR(128) NOT NULL,
    record_id UUID NOT NULL,
    content TEXT NOT NULL,
    status VARCHAR(64) DEFAULT 'pending', -- pending, processing, completed, failed
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE
);
CREATE INDEX IF NOT EXISTS embedding_queue_status_idx ON embedding_queue(status);

-- Trigger to queue embedding generation (separate functions per table to avoid field access errors)
CREATE OR REPLACE FUNCTION queue_ontology_embedding()
RETURNS TRIGGER AS $$
BEGIN
    -- Use description if available, otherwise fall back to content
    IF NEW.embedding IS NULL AND (NEW.description IS NOT NULL OR NEW.content IS NOT NULL) THEN
        INSERT INTO embedding_queue (table_name, record_id, content)
        VALUES ('ontologies', NEW.id, COALESCE(NEW.name || ': ', '') || COALESCE(NEW.description, NEW.content));
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION queue_resource_embedding()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.embedding IS NULL AND NEW.content IS NOT NULL THEN
        INSERT INTO embedding_queue (table_name, record_id, content)
        VALUES ('resources', NEW.id, COALESCE(NEW.name || ': ', '') || NEW.content);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION queue_message_embedding()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.embedding IS NULL AND NEW.content IS NOT NULL THEN
        INSERT INTO embedding_queue (table_name, record_id, content)
        VALUES ('messages', NEW.id, NEW.content);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS ontologies_embedding_queue ON ontologies;
CREATE TRIGGER ontologies_embedding_queue
AFTER INSERT OR UPDATE OF description, content ON ontologies
FOR EACH ROW EXECUTE FUNCTION queue_ontology_embedding();

DROP TRIGGER IF EXISTS resources_embedding_queue ON resources;
CREATE TRIGGER resources_embedding_queue
AFTER INSERT OR UPDATE OF content ON resources
FOR EACH ROW EXECUTE FUNCTION queue_resource_embedding();

DROP TRIGGER IF EXISTS messages_embedding_queue ON messages;
CREATE TRIGGER messages_embedding_queue
AFTER INSERT OR UPDATE OF content ON messages
FOR EACH ROW EXECUTE FUNCTION queue_message_embedding();

-- Scenario embedding queue
CREATE OR REPLACE FUNCTION queue_scenario_embedding()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.embedding IS NULL AND NEW.description IS NOT NULL THEN
        INSERT INTO embedding_queue (table_name, record_id, content)
        VALUES ('scenarios', NEW.id, COALESCE(NEW.name || ': ', '') || NEW.description);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS scenarios_embedding_queue ON scenarios;
CREATE TRIGGER scenarios_embedding_queue
AFTER INSERT OR UPDATE OF description ON scenarios
FOR EACH ROW EXECUTE FUNCTION queue_scenario_embedding();

-- Collection embedding queue
CREATE OR REPLACE FUNCTION queue_collection_embedding()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.embedding IS NULL AND NEW.description IS NOT NULL THEN
        INSERT INTO embedding_queue (table_name, record_id, content)
        VALUES ('collections', NEW.id, COALESCE(NEW.name || ': ', '') || NEW.description);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS collections_embedding_queue ON collections;
CREATE TRIGGER collections_embedding_queue
AFTER INSERT OR UPDATE OF description ON collections
FOR EACH ROW EXECUTE FUNCTION queue_collection_embedding();

-- Agent embedding queue
-- Uses description if provided, otherwise falls back to content for embedding
CREATE OR REPLACE FUNCTION queue_agent_embedding()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.embedding IS NULL AND (NEW.description IS NOT NULL OR NEW.content IS NOT NULL) THEN
        INSERT INTO embedding_queue (table_name, record_id, content)
        VALUES ('agents', NEW.id, COALESCE(NEW.name || ': ', '') || COALESCE(NEW.description, NEW.content));
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS agents_embedding_queue ON agents;
CREATE TRIGGER agents_embedding_queue
AFTER INSERT OR UPDATE OF description, content ON agents
FOR EACH ROW EXECUTE FUNCTION queue_agent_embedding();

-- Server embedding queue
CREATE OR REPLACE FUNCTION queue_server_embedding()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.embedding IS NULL AND NEW.description IS NOT NULL THEN
        INSERT INTO embedding_queue (table_name, record_id, content)
        VALUES ('servers', NEW.id, COALESCE(NEW.name || ': ', '') || NEW.description);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS servers_embedding_queue ON servers;
CREATE TRIGGER servers_embedding_queue
AFTER INSERT OR UPDATE OF description ON servers
FOR EACH ROW EXECUTE FUNCTION queue_server_embedding();

-- Tool embedding queue
CREATE OR REPLACE FUNCTION queue_tool_embedding()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.embedding IS NULL AND NEW.description IS NOT NULL THEN
        INSERT INTO embedding_queue (table_name, record_id, content)
        VALUES ('tools', NEW.id, COALESCE(NEW.name || ': ', '') || NEW.description);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tools_embedding_queue ON tools;
CREATE TRIGGER tools_embedding_queue
AFTER INSERT OR UPDATE OF description ON tools
FOR EACH ROW EXECUTE FUNCTION queue_tool_embedding();

-- ============================================
-- SEED DATA (for testing)
-- ============================================

-- Test user
INSERT INTO users (id, name, email, summary, interests, preferred_topics, activity_level, user_id)
VALUES (
    'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11',
    'Test User',
    'test@example.com',
    'A test user for development and integration testing. Interested in AI, machine learning, and software development.',
    ARRAY['artificial intelligence', 'machine learning', 'software development', 'data science'],
    ARRAY['agents', 'LLMs', 'RAG', 'knowledge graphs'],
    'active',
    'test-user'
)
ON CONFLICT (email) DO UPDATE SET
    name = EXCLUDED.name,
    summary = EXCLUDED.summary,
    interests = EXCLUDED.interests,
    preferred_topics = EXCLUDED.preferred_topics,
    updated_at = NOW();

-- Test projects (stored in ontologies table with entity_type = 'project')
INSERT INTO ontologies (id, name, description, category, entity_type, properties, tags)
VALUES
    (
        'b1eebc99-9c0b-4ef8-bb6d-6bb9bd380b22',
        'project-alpha',
        'Project Alpha is a machine learning pipeline for automated document processing.',
        'engineering',
        'project',
        '{"status": "active", "lead": "sarah-chen", "team_size": 5, "start_date": "2024-01-15", "budget": 150000, "priority": "high"}',
        ARRAY['ml', 'documents', 'automation']
    ),
    (
        'c2eebc99-9c0b-4ef8-bb6d-6bb9bd380c33',
        'project-beta',
        'Project Beta focuses on building a real-time analytics dashboard for business intelligence.',
        'data',
        'project',
        '{"status": "planning", "lead": "john-doe", "team_size": 3, "start_date": "2024-03-01", "budget": 80000, "priority": "medium"}',
        ARRAY['analytics', 'dashboard', 'bi']
    ),
    (
        'd3eebc99-9c0b-4ef8-bb6d-6bb9bd380d44',
        'project-gamma',
        'Project Gamma is an AI-powered customer support chatbot using RAG architecture.',
        'ai',
        'project',
        '{"status": "active", "lead": "jane-smith", "team_size": 4, "start_date": "2024-02-10", "budget": 200000, "priority": "high"}',
        ARRAY['ai', 'chatbot', 'rag', 'support']
    )
ON CONFLICT (name) DO UPDATE SET
    description = EXCLUDED.description,
    category = EXCLUDED.category,
    entity_type = EXCLUDED.entity_type,
    properties = EXCLUDED.properties,
    tags = EXCLUDED.tags,
    updated_at = NOW();

-- ============================================
-- DONE
-- ============================================
SELECT 'REMLight database installed successfully (with seed data)' as status;
