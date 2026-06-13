-- AI Content Studio — Row Level Security policies
-- Run this in the Supabase SQL editor AFTER schema.sql.
--
-- These are PERMISSIVE policies suitable for a personal demo: they let the
-- publishable/anon key read and write both tables. For a multi-user / public
-- production app you'd scope rows by auth.uid() instead.

alter table public.generations    enable row level security;
alter table public.brand_examples enable row level security;

-- generations: open read + write for the demo
drop policy if exists demo_all_generations on public.generations;
create policy demo_all_generations
    on public.generations
    for all
    using (true)
    with check (true);

-- brand_examples: open read + write for the demo
drop policy if exists demo_all_brand_examples on public.brand_examples;
create policy demo_all_brand_examples
    on public.brand_examples
    for all
    using (true)
    with check (true);
