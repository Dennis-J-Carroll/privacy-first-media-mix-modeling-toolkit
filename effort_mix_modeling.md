# Effort Mix Modeling Guidelines

## Conceptual Shift: From Corporate MMM to Personal Effort Mix Modeling
For individual creators, influencers or small businesses, the most valuable resource isn’t always money—it’s time and creative energy. Effort Mix Modeling reframes media mix modeling so that the “spend” variable captures the effort you put into creating and promoting content.

### Redefining the “Spend” (input variables)
- **Monetary spend**: Small promotional budgets and subscriptions (e.g. boosting a post for $50, a $100/week ad campaign or paying for a scheduling tool).
- **Time and effort spend**: Units of work tracked per channel, such as:
  - Number of TikTok videos, Instagram reels or YouTube videos posted.
  - Hours spent editing videos or engaging with your audience.
  - A subjective “effort score” (1–5) capturing how labor‑intensive each piece of content was.
- **Promotional activities**: Flags for one-off actions such as running a sale, sending a newsletter or collaborating with another creator.

### Redefining the “Revenue” (goal variables)
Rather than just revenue, model whatever goal is most important at a given time. Run separate models for different goals:
- **Audience growth**: weekly new followers or subscribers.
- **Engagement**: total weekly comments, likes or shares.
- **Lead generation**: signups to a newsletter or link clicks.
- **Direct sales**: weekly revenue or number of sales if you sell products.

### Adstock (carryover) and Saturation in the personal context
- **Adstock/carryover**: Content continues to work after it’s posted. Stories decay quickly (low adstock); viral reels last several days; evergreen YouTube videos have high adstock and continue to drive views for months.
- **Saturation/diminishing returns**: Too much output can cause audience fatigue. Posting one great reel a day might be optimal; posting five per day may reduce engagement per post. Effort Mix Modeling helps find the sweet spot.

## Practical Plan for Using Effort Mix Modeling
1. **Track your data**: Maintain a simple weekly spreadsheet that records your goal metrics and effort inputs. For example:

   | Week Start Date | New Followers | Link Clicks | Revenue | Instagram Reels Posted | Hours on TikTok | Boosted Post Spend | Newsletter Sent? |
   | --- | --- | --- | --- | --- | --- | --- | --- |
   | 2025‑09‑01 | 150 | 320 | $450 | 4 | 5 | $25 | 1 |
   | 2025‑09‑08 | 175 | 410 | $520 | 5 | 6 | $50 | 0 |
   | 2025‑09‑15 | 160 | 380 | $480 | 4 | 4 | $0 | 1 |

   Aim for at least 15–20 weeks of consistent data before fitting a model.

2. **Fit a simplified model**: For small datasets, use linear regression on transformed variables rather than a complex non-linear optimizer. Apply adstock to effort variables and a logarithmic or similar transformation to model saturation. Python libraries such as pandas and statsmodels make this accessible.

3. **Interpret results**: Visualize contributions of each effort variable to your goal and compute a return on effort (e.g. followers per reel or revenue per hour spent). Adjust your strategy based on which activities generate the best returns.

Effort Mix Modeling helps creators understand where to focus their limited time and resources for maximum impact.
