-- AI Content Studio — Supabase schema
-- Run this in the Supabase SQL editor (Dashboard → SQL → New query).
-- Embedding dimension 384 matches sentence-transformers/all-MiniLM-L6-v2.

-- 1) Enable pgvector
create extension if not exists vector;

-- 2) Generation history (so the app "doesn't forget")
create table if not exists generations (
    id          bigint generated always as identity primary key,
    domain      text        not null,
    topic       text        not null,
    channel     text,
    tone        text,
    items       jsonb       not null,      -- list of generated items + brand_match
    created_at  timestamptz not null default now()
);
create index if not exists generations_created_at_idx
    on generations (created_at desc);

-- 3) Brand / knowledge examples for RAG
create table if not exists brand_examples (
    id          bigint generated always as identity primary key,
    content     text         not null,
    metadata    jsonb        not null default '{}'::jsonb,
    embedding   vector(384),
    created_at  timestamptz  not null default now()
);

-- Approximate-NN index for fast similarity search
create index if not exists brand_examples_embedding_idx
    on brand_examples using ivfflat (embedding vector_cosine_ops) with (lists = 100);

-- 4) Similarity search RPC used by SupabaseVectorStore.retrieve()
create or replace function match_brand_examples (
    query_embedding vector(384),
    match_count     int default 3,
    filter_domain   text default null
)
returns table (id bigint, content text, metadata jsonb, similarity float)
language sql stable as $$
    select  e.id,
            e.content,
            e.metadata,
            1 - (e.embedding <=> query_embedding) as similarity
    from    brand_examples e
    where   e.embedding is not null
      and   (filter_domain is null
             or e.metadata->>'domain' is null
             or e.metadata->>'domain' = filter_domain)
    order by e.embedding <=> query_embedding
    limit   match_count;
$$;

-- Row Level Security: new projects enable RLS, so the publishable/anon key
-- needs policies before it can read/write. See supabase/policies.sql and run
-- it after this file (kept separate so you can tighten it for production).
