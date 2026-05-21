# Design Notes

> Stub — decision log populated in Sprint 8.

## Future Work

### Multi-tenant ACL mirroring

In a shared-workspace scenario (e.g., a team Drive), each user should only retrieve
chunks they have Drive read permission for. The correct implementation mirrors Drive's
ACL at indexing time: each chunk's metadata carries the `permissionIds` of accounts
with at least reader access, and the vector query is filtered to chunks where the
authenticated user's ID is in that set.

This is out of scope for v1 (single-user, local-first) but is the right design when
moving to a multi-user deployment. The alternative — post-query ACL filtering — is
insecure because it leaks chunk count and rank signal to unauthorized users.
