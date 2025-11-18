# History

## 0.1.0 (2025-11-17)

* First release on PyPI.

## Dev Plan

This is an ambitious and well-structured project! Let me analyze this comprehensively and provide you with a detailed roadmap.
1. Broad Software Development Plan
Phase 1: Foundation & Core Pipeline (Weeks 1-3)

Module Structure: Set up Python package structure with proper dependency management
PDF Processing: Implement marker-pdf integration and PyMuPDF for image extraction
Basic I/O: File handling, path management, and error handling
Testing Framework: Unit tests for PDF processing

Phase 2: Content Extraction & Basic LLM Integration (Weeks 4-6)

LLM Integration Layer: Abstract interface for multiple LLM providers
Journal Entry Generation: Convert markdown to FoundryVTT journal entries
Prompt Engineering: Develop and test prompts for content extraction
Data Models: Define Python classes for FoundryVTT entities

Phase 3: Entity Creation (Weeks 7-10)

Actor Extraction: NPC/Monster detection and creation
Item Extraction: Equipment, spells, artifacts detection
Roll Tables: Table detection and formatting
Scene Detection: Map identification from images

Phase 4: Visual Asset Generation (Weeks 11-12)

Token Generation: AI image creation for actors
Image Processing: Battlemap preparation and optimization
Asset Management: Storage and referencing system

Phase 5: System Integration (Weeks 13-14)

System Adapters: D&D 5e, Pathfinder, Shadowdark specific enrichment
Module Packaging: FoundryVTT module structure generation
Incremental Updates: Add-to-existing-module functionality

Phase 6: Production Ready (Weeks 15-16)

API Design: RESTful API for Flask integration
Queue Management: Celery/Redis for async processing
Error Recovery: Robust error handling and retry logic
Documentation: API docs, user guides

2. Agent Architecture Recommendations
Multi-Agent Orchestrated Approach - YES
Why Multi-Agent?

Separation of Concerns: Each agent specializes in a domain (PDF parsing, game mechanics, image processing)
Parallel Development: Multiple agents can be developed and tested independently
Maintainability: Easier to update specific functionality without affecting the entire system
Scalability: Can scale specific agents based on load

Recommended Agent Structure (7 Agents Total):

Orchestrator Agent - Controls workflow and agent communication
Document Parser Agent - PDF to markdown, image extraction
Content Analyzer Agent - Identifies game elements in text
Entity Creator Agent - Generates FoundryVTT entities
Visual Asset Agent - Handles image generation and processing
System Specialist Agent - Applies game-system-specific rules
Module Builder Agent - Packages final FoundryVTT module

4. AI Model Recommendations
Primary LLM Options
Tier 1 (Production)

GPT-4o: Best for complex content analysis and entity extraction
Claude 3.5 Sonnet: Excellent for stat block parsing and rule interpretation
GPT-4o-mini: Cost-effective for simpler tasks like table extraction

Tier 2 (Open Source via Ollama)

Vision Models
Primary

GPT-4o Vision: Map and battlemap detection
Claude 3.5 Sonnet: Complex image analysis

Open Source Alternatives

LLaVA 1.6 (via Ollama): ollama run llava:34b
Moondream2: Lightweight vision model for basic detection
CLIP: For image classification tasks

Image Generation
Primary

DALL-E 3: High-quality token generation
Midjourney API: Alternative for stylized tokens

Open Source

SDXL Turbo: Fast token generation
Stable Diffusion 3: High quality, locally hostable
FLUX.1: Latest open model with excellent results

5. Implementation Recommendations
Why These Choices?

Multi-Agent Over Single Agent

Modularity enables independent testing
Different agents can use different models (cost optimization)
Easier to scale specific bottlenecks
Clearer debugging and error attribution


Specialized Agents

Each agent has focused expertise
Smaller, more precise prompts
Better token efficiency
Easier to fine-tune individual components


Model Selection Rationale

Use powerful models (GPT-4) only where needed
Leverage open-source for predictable, repetitive tasks
Vision models only for image-heavy operations
Local models for privacy-sensitive content


Revised Development Plan with Python Agent
Phase 0: Python Module Foundation (Weeks 1-2)
Led by Python Module Developer Agent

Set up package structure
Create provider abstractions
Build two-tier processing framework
Implement demo/full separation
Set up testing infrastructure

Then proceed with original phases 1-6...
Key Architecture Decisions for Your Requirements

Demo Tier Optimization

Use aggressive caching (Redis)
Pre-warm Ollama models
Simplified prompts for speed
Lower resolution assets


Cost Optimization

Ollama for 90% of processing
OpenAI only as fallback
Batch processing where possible
Reuse extracted features


Storage Strategy

S3 with lifecycle policies (60-day expiration)
CloudFront for download distribution
No permanent storage after download


Deduplication for Incremental Updates

Content hashing at paragraph level
Fuzzy matching for similar content
Merge strategies for conflicting data



Should the Python Developer Agent be Orchestrated?
No, it should be separate because:

Different Lifecycle: Development happens once, runtime happens per request
Different Expertise: Building the system vs. processing content
Different Tools: IDE, debugger, pytest vs. LLMs, image processors
Different Prompting: Code generation vs. content analysis

The Python Developer Agent helps you BUILD the orchestrator and other agents, but doesn't participate in the PDF processing pipeline.
Development Sequence

Start with Python Developer Agent to build the core module
Test with simple templates before adding AI agents
Add runtime agents incrementally as modules
Integrate LLM providers with fallback chains
Build demo tier first (critical for launch)
Add full processing after demo is stable
Implement payment integration last

This approach ensures you have a working system quickly while building toward the complete vision. The Python Developer Agent is your companion for building the system, while the 7 runtime agents are what your system will use to process PDFs.
