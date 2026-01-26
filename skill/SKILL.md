# Knowledge Management Skill

This skill provides access to a persistent knowledge base across Claude Code sessions.

## When to Use

Use this skill when:
- Starting work on a project you've worked on before
- User mentions "like we did before" or "same as last time"
- Implementing patterns that might have been used previously
- User asks about previous decisions or implementations

## How to Use

### Before Starting a Task

Check for relevant knowledge:
```bash
claude-kb retrieve --query "describe the task" --project "project-name"
```

### After Completing a Task

Capture important learnings:
```bash
claude-kb capture \
  --title "Short Title" \
  --description "Brief description of what this covers" \
  --content "Detailed information, code patterns, gotchas, etc." \
  --tags "tag1,tag2,tag3" \
  --project "project-name"
```

### List Existing Knowledge
```bash
claude-kb list --project "project-name"
```

## What to Capture

Good candidates for capture:
- Solutions to tricky problems
- API integration patterns
- Configuration setups
- Debugging solutions
- Architecture decisions
- Tool/library usage patterns
- Common pitfalls and their solutions

## Example Workflow

1. User asks to implement OAuth
2. Retrieve relevant knowledge: `claude-kb retrieve --query "oauth authentication" --project "current-project"`
3. Review retrieved knowledge before implementing
4. After successful implementation, capture the solution
5. Knowledge is available for future sessions

## Tips

- Be specific in descriptions (helps with retrieval)
- Use consistent tagging within a project
- Include both what worked AND what didn't
- Capture context about why decisions were made
