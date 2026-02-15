/**
 * Literature Assignment Widget
 *
 * Manages curator assignment UI for literature review reports.
 * Reads configuration from embedded JSON blocks and renders:
 * 1. ToC highlighting: CSS classes on gene links in the Table of Contents
 * 2. Header widgets: Assignment dropdowns in .assignment-slot elements
 *
 * All gene keying uses HGNC IDs (e.g. "HGNC:8607"). Gene symbols for
 * display are looked up via reportConfig.genes[hgncId].hgnc_symbol.
 */
var LiteratureAssignments = {
  config: null,
  completionStatus: null,
  _tocLinkMap: null,

  /**
   * Escape HTML special characters to prevent XSS.
   */
  _escapeHtml: function (text) {
    var div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  },

  /**
   * Look up the display symbol for an HGNC ID from report config.
   */
  _getGeneSymbol: function (hgncId) {
    return this.config.reportConfig.genes[hgncId].hgnc_symbol;
  },

  /**
   * Build tooltip text for a skipped assignment.
   */
  _skipTooltip: function (assignment) {
    var skippedType = assignment.assigned_to ? "Investigated" : "Triaged";
    var skippedBy = this._getSkippedByName(assignment);
    if (skippedBy) {
      return skippedType + " by " + skippedBy + ": " + assignment.skipped_reason;
    }
    return skippedType + ": " + assignment.skipped_reason;
  },

  /**
   * Get curator display name (initials) from user ID.
   */
  _getCuratorName: function (userId) {
    if (!userId) return null;
    for (var i = 0; i < this.config.curators.length; i++) {
      if (this.config.curators[i].id === userId) {
        return this.config.curators[i].initials;
      }
    }
    return null;
  },

  /**
   * Get skipped_by display name (initials) from assignment.
   */
  _getSkippedByName: function (assignment) {
    if (!assignment || !assignment.skipped_by) return null;
    for (var i = 0; i < this.config.curators.length; i++) {
      if (this.config.curators[i].id === assignment.skipped_by) {
        return this.config.curators[i].initials;
      }
    }
    return null;
  },

  /**
   * Build tooltip text for an assigned (in-progress) item.
   */
  _assignedTooltip: function (assignment) {
    var name = this._getCuratorName(assignment.assigned_to);
    return name ? "Assigned to " + name : "Assigned";
  },

  /**
   * Build tooltip text for a concordant completion.
   */
  _concordantTooltip: function (assignment) {
    var name = this._getCuratorName(assignment ? assignment.assigned_to : null);
    return name ? "Completed by " + name : null;
  },

  /**
   * Build tooltip text for a discordant completion.
   */
  _discordantTooltip: function (completion, assignment) {
    var actual = completion.ratings.join(", ");
    var suggested = completion.suggestedRating;
    var name = this._getCuratorName(assignment ? assignment.assigned_to : null);
    if (name) {
      return "Rated " + actual + " by " + name + " (suggested " + suggested + ")";
    }
    return "Rated " + actual + " (suggested " + suggested + ")";
  },

  /**
   * Set instant tooltip on an element using data-title attribute.
   */
  _setTooltip: function (element, text) {
    if (text) {
      element.setAttribute("data-title", text);
    } else {
      element.removeAttribute("data-title");
    }
  },

  init: function () {
    // Read configuration from both JSON blocks
    var reportConfig = this._readJsonBlock("report-config");
    var assignmentData = this._readJsonBlock("assignment-data");

    // Merge into unified config
    this.config = {
      reportId: assignmentData.reportId,
      reportConfig: reportConfig,
      currentUserId: assignmentData.currentUserId,
      curators: assignmentData.curators,
      assignments: assignmentData.assignments,
      completions: assignmentData.completions,
    };

    // Validate required config (fail-fast)
    if (!this.config.reportId)
      throw new Error("assignmentData.reportId is required");
    if (!this.config.reportConfig) throw new Error("reportConfig is required");
    if (!this.config.curators)
      throw new Error("assignmentData.curators is required");
    if (!this.config.assignments)
      throw new Error("assignmentData.assignments is required");
    if (!this.config.completions)
      throw new Error("assignmentData.completions is required");
    if (this.config.currentUserId === undefined)
      throw new Error("assignmentData.currentUserId is required");

    this.completionStatus = this._processCompletions(this.config.completions);
    this._buildTocLinkMap();
    this.highlightTocLinks();
    this.renderAllWidgets();
  },

  _readJsonBlock: function (id) {
    var el = document.getElementById(id);
    if (!el) throw new Error("Missing required JSON block: #" + id);
    try {
      return JSON.parse(el.textContent);
    } catch (e) {
      throw new Error("Invalid JSON in #" + id + ": " + e.message);
    }
  },

  /**
   * Build a lookup map of ToC links by HGNC ID (called once at init).
   * This avoids repeated querySelector calls in highlightTocLinks.
   */
  _buildTocLinkMap: function () {
    this._tocLinkMap = {};
    var links = document.querySelectorAll(
      '.toc-gene-list a[href^="#novel-gene-"], .toc-gene-list a[href^="#known-gene-"]'
    );
    for (var i = 0; i < links.length; i++) {
      var href = links[i].getAttribute("href");
      var match = href.match(/#(?:novel|known)-gene-(.+)$/);
      if (match) {
        this._tocLinkMap["HGNC:" + match[1]] = links[i];
      }
    }
  },

  /**
   * Find ToC link for a gene using the pre-built lookup map.
   */
  _findTocLink: function (hgncId) {
    return this._tocLinkMap[hgncId] || null;
  },

  /**
   * Apply status class and tooltip to a ToC link based on assignment/completion.
   * Priority: completion > assigned > skipped.
   */
  _applyTocLinkStatus: function (link, hgncId) {
    var assignment = this.config.assignments[hgncId];
    var completion = this.completionStatus[hgncId];

    if (completion && completion.completed) {
      if (completion.concordant) {
        link.classList.add("concordance-concordant");
        this._setTooltip(link, this._concordantTooltip(assignment));
      } else {
        link.classList.add("concordance-discordant");
        this._setTooltip(link, this._discordantTooltip(completion, assignment));
      }
    } else if (assignment && assignment.status === "skipped") {
      link.classList.add("concordance-not_executed");
      this._setTooltip(link, this._skipTooltip(assignment));
    } else if (assignment && assignment.status === "assigned") {
      if (assignment.assigned_to === this.config.currentUserId) {
        link.classList.add("my-task");
        this._setTooltip(link, "Assigned to you");
      } else {
        link.classList.add("assigned-other");
        this._setTooltip(link, this._assignedTooltip(assignment));
      }
    }
  },

  /**
   * Apply CSS classes to ToC links based on assignment/completion status.
   * Uses same classes as palit/templates/concordance_report_styles.css.
   */
  highlightTocLinks: function () {
    var self = this;
    var genes = this.config.reportConfig.genes;
    if (!genes) throw new Error("reportConfig.genes is required");

    Object.keys(genes).forEach(function (hgncId) {
      var link = self._findTocLink(hgncId);
      if (!link) return; // Gene not in ToC (might be filtered out)
      self._applyTocLinkStatus(link, hgncId);
    });
  },

  _processCompletions: function (completions) {
    var result = {};
    var genes = this.config.reportConfig.genes;
    if (!genes) throw new Error("reportConfig.genes is required");

    Object.keys(completions).forEach(function (hgncId) {
      var data = completions[hgncId];
      if (data.has_evaluation) {
        var geneConfig = genes[hgncId];
        if (!geneConfig)
          throw new Error("Gene " + hgncId + " not found in reportConfig.genes");
        if (!geneConfig.suggested_rating)
          throw new Error("Gene " + hgncId + " missing suggested_rating");

        // Concordant if ANY rating matches the suggestion
        var concordant = data.ratings.indexOf(geneConfig.suggested_rating) >= 0;

        result[hgncId] = {
          completed: true,
          ratings: data.ratings,
          suggestedRating: geneConfig.suggested_rating,
          concordant: concordant,
        };
      }
    });

    return result;
  },

  renderAllWidgets: function () {
    var self = this;
    document.querySelectorAll(".assignment-slot").forEach(function (slot) {
      var hgncId = slot.dataset.hgncId;
      var assignment = self.config.assignments[hgncId];
      var completion = self.completionStatus[hgncId];
      slot.innerHTML = self.renderWidget(hgncId, assignment, completion);
      self.bindEvents(slot, hgncId, assignment);
    });
  },

  renderWidget: function (hgncId, assignment, completion) {
    var status = assignment ? assignment.status : "pending";

    // Completed state: no widget needed (status shown in ToC)
    if (completion && completion.completed) {
      return "";
    }

    // Skipped state: show badge with reason and restore button
    if (status === "skipped") {
      var skipTitle = this._escapeHtml(this._skipTooltip(assignment));
      return (
        '<span class="lit-widget lit-skipped-widget">' +
        '<span class="lit-badge lit-skipped" data-title="' +
        skipTitle +
        '">&mdash;</span>' +
        '<button class="lit-restore-btn" data-hgnc-id="' +
        hgncId +
        '" data-title="Restore">&#x21ba;</button>' +
        "</span>"
      );
    }

    // Pending or Assigned: show dropdown + skip button
    var currentAssignee = assignment ? assignment.assigned_to : null;
    var widgetClass = "";
    if (status === "assigned" && currentAssignee) {
      // Differentiate between "my task" (coral) vs "someone else's" (teal)
      widgetClass = currentAssignee === this.config.currentUserId
        ? "lit-assigned-mine"
        : "lit-assigned-other";
    }

    var html =
      '<span class="lit-widget ' +
      widgetClass +
      '">' +
      '<select class="lit-assignee-select" autocomplete="off" name="assignee-' +
      hgncId +
      '" data-hgnc-id="' +
      hgncId +
      '" data-current="' +
      (currentAssignee || "") +
      '">' +
      '<option value="">\u2014</option>' +
      this.config.curators
        .map(function (u) {
          var selected = u.id === currentAssignee ? " selected" : "";
          return (
            '<option value="' +
            u.id +
            '"' +
            selected +
            ">" +
            u.initials +
            "</option>"
          );
        })
        .join("") +
      "</select>" +
      '<button class="lit-skip-btn" data-hgnc-id="' +
      hgncId +
      '" data-title="Skip">&times;</button>' +
      "</span>";

    return html;
  },

  bindEvents: function (slot, hgncId, assignment) {
    var self = this;
    var select = slot.querySelector(".lit-assignee-select");
    if (select) {
      select.addEventListener("change", function (e) {
        self.handleAssign(e, hgncId, assignment);
      });
    }

    var skipBtn = slot.querySelector(".lit-skip-btn");
    if (skipBtn) {
      skipBtn.addEventListener("click", function (e) {
        self.handleSkip(e, hgncId, assignment);
      });
    }

    var restoreBtn = slot.querySelector(".lit-restore-btn");
    if (restoreBtn) {
      restoreBtn.addEventListener("click", function (e) {
        self.handleRestore(e, hgncId, assignment);
      });
    }
  },

  handleAssign: function (event, hgncId, currentAssignment) {
    var self = this;
    var select = event.target;
    var newAssigneeId = select.value ? parseInt(select.value) : null;

    // For optimistic locking: null means we expect record doesn't exist
    var expectedUpdatedAt = currentAssignment
      ? currentAssignment.updated_at
      : null;

    fetch("/api/v1/literature-assignments/assign/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": this.getCsrfToken(),
      },
      body: JSON.stringify({
        report_id: this.config.reportId,
        hgnc_id: hgncId,
        assigned_to: newAssigneeId,
        expected_updated_at: expectedUpdatedAt,
      }),
    })
      .then(function (resp) {
        if (resp.status === 409) {
          return resp.json().then(function (data) {
            var state = data.current_state;
            var currentHolder = state.assigned_to_display || "someone else";
            alert(
              "This gene was just updated by another user.\n\n" +
                "Current status: " +
                state.status +
                "\n" +
                "Assigned to: " +
                currentHolder +
                "\n\n" +
                "The dropdown has been refreshed. Please try again."
            );
            self.config.assignments[hgncId] = Object.assign(
              {},
              self.config.assignments[hgncId],
              state
            );
            self.refreshGene(hgncId);
          });
        }

        if (resp.status === 404) {
          return resp.json().then(function () {
            var symbol = self._getGeneSymbol(hgncId);
            alert(
              "Gene not found: " +
                symbol +
                " (" +
                hgncId +
                ")\n\n" +
                "This gene doesn't exist in the PanelApp database. " +
                "Please contact an administrator."
            );
          });
        }

        if (!resp.ok) {
          return resp.json().then(function (data) {
            alert("Failed to assign: " + (data.message || resp.status));
            self.refreshGene(hgncId);
          });
        }

        return resp.json().then(function (updated) {
          self.config.assignments[hgncId] = updated;
          delete self.completionStatus[hgncId];
          self.refreshGene(hgncId);
        });
      })
      .catch(function (err) {
        console.error("Assignment failed:", err);
        alert(
          "Network error while updating assignment. Please check your connection and try again."
        );
        self.refreshGene(hgncId);
      });
  },

  handleSkip: function (event, hgncId, currentAssignment) {
    var self = this;
    var symbol = this._getGeneSymbol(hgncId);
    var reason = prompt(
      "Why are you skipping " +
        symbol +
        "?\n\n" +
        "Common reasons:\n" +
        "- Review already added\n" +
        "- Out of scope (e.g., cancer gene)\n" +
        "- Unconvincing evidence / poor quality source\n" +
        "- Nuanced clinical assessment not captured by LLM"
    );

    if (!reason || !reason.trim()) {
      return;
    }

    // For optimistic locking: null means we expect record doesn't exist
    var expectedUpdatedAt = currentAssignment
      ? currentAssignment.updated_at
      : null;

    fetch("/api/v1/literature-assignments/skip/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": this.getCsrfToken(),
      },
      body: JSON.stringify({
        report_id: this.config.reportId,
        hgnc_id: hgncId,
        reason: reason.trim(),
        expected_updated_at: expectedUpdatedAt,
      }),
    })
      .then(function (resp) {
        if (resp.status === 409) {
          return resp.json().then(function (data) {
            var state = data.current_state;
            var currentHolder = state.assigned_to_display || "unassigned";
            var msg =
              "This gene was just updated by another user.\n\n" +
              "Current status: " +
              state.status +
              "\n" +
              "Assigned to: " +
              currentHolder +
              "\n";
            if (state.skipped_reason) {
              msg += "Skip reason: " + state.skipped_reason + "\n";
            }
            msg +=
              "\nThe widget has been refreshed. Please try again if needed.";
            alert(msg);
            self.config.assignments[hgncId] = Object.assign(
              {},
              self.config.assignments[hgncId],
              state
            );
            self.refreshGene(hgncId);
          });
        }

        if (resp.status === 404) {
          return resp.json().then(function () {
            var symbol = self._getGeneSymbol(hgncId);
            alert(
              "Gene not found: " +
                symbol +
                " (" +
                hgncId +
                ")\n\n" +
                "This gene doesn't exist in the PanelApp database. " +
                "Please contact an administrator."
            );
          });
        }

        if (!resp.ok) {
          return resp.json().then(function (data) {
            alert("Failed to skip: " + (data.message || resp.status));
          });
        }

        return resp.json().then(function (updated) {
          self.config.assignments[hgncId] = updated;
          self.refreshGene(hgncId);
        });
      })
      .catch(function (err) {
        console.error("Skip failed:", err);
        alert(
          "Network error while skipping. Please check your connection and try again."
        );
      });
  },

  handleRestore: function (event, hgncId, currentAssignment) {
    var self = this;

    // Restore by calling assign with null assignee (sets to pending)
    var expectedUpdatedAt = currentAssignment
      ? currentAssignment.updated_at
      : null;

    fetch("/api/v1/literature-assignments/assign/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": this.getCsrfToken(),
      },
      body: JSON.stringify({
        report_id: this.config.reportId,
        hgnc_id: hgncId,
        assigned_to: null,
        expected_updated_at: expectedUpdatedAt,
      }),
    })
      .then(function (resp) {
        if (resp.status === 409) {
          return resp.json().then(function (data) {
            var state = data.current_state;
            alert(
              "This gene was just updated by another user.\n\n" +
                "Current status: " +
                state.status +
                "\n\n" +
                "The widget has been refreshed."
            );
            self.config.assignments[hgncId] = Object.assign(
              {},
              self.config.assignments[hgncId],
              state
            );
            self.refreshGene(hgncId);
          });
        }

        if (!resp.ok) {
          return resp.json().then(function (data) {
            alert("Failed to restore: " + (data.message || resp.status));
          });
        }

        return resp.json().then(function (updated) {
          self.config.assignments[hgncId] = updated;
          self.refreshGene(hgncId);
        });
      })
      .catch(function (err) {
        console.error("Restore failed:", err);
        alert(
          "Network error while restoring. Please check your connection and try again."
        );
      });
  },

  /**
   * Refresh both widget and ToC link for a gene after status change.
   */
  refreshGene: function (hgncId) {
    // Refresh widget
    var slot = document.querySelector(
      '.assignment-slot[data-hgnc-id="' + hgncId + '"]'
    );
    if (slot) {
      var assignment = this.config.assignments[hgncId];
      var completion = this.completionStatus[hgncId];
      slot.innerHTML = this.renderWidget(hgncId, assignment, completion);
      this.bindEvents(slot, hgncId, assignment);
    }

    // Refresh ToC link
    var link = this._findTocLink(hgncId);
    if (link) {
      // Remove all status classes and tooltip, then re-apply
      link.classList.remove(
        "my-task",
        "assigned-other",
        "concordance-concordant",
        "concordance-discordant",
        "concordance-not_executed"
      );
      this._setTooltip(link, null);
      this._applyTocLinkStatus(link, hgncId);
    }
  },

  getCsrfToken: function () {
    var tokenEl = document.querySelector("[name=csrfmiddlewaretoken]");
    if (tokenEl) return tokenEl.value;
    var match = document.cookie.match(/csrftoken=([^;]+)/);
    return match ? match[1] : "";
  },
};

/**
 * Handle prefill button clicks by dynamically creating and submitting a form.
 * This avoids embedding thousands of hidden inputs in the HTML, which slows
 * down page load due to browser autofill scanning.
 */
function handlePrefillClick(event) {
  var button = event.currentTarget;
  var prefillData = JSON.parse(button.dataset.prefill);

  // Create form dynamically
  var form = document.createElement("form");
  form.method = "POST";
  form.action = "/panels/reports/prefill/";
  form.target = "_blank";

  // Add CSRF token
  var csrfToken =
    document.querySelector("[name=csrfmiddlewaretoken]")?.value ||
    document.cookie.match(/csrftoken=([^;]+)/)?.[1] ||
    "";
  var csrfInput = document.createElement("input");
  csrfInput.type = "hidden";
  csrfInput.name = "csrfmiddlewaretoken";
  csrfInput.value = csrfToken;
  form.appendChild(csrfInput);

  // Map prefill data to form fields
  var fields = {
    form_type: prefillData.form_type,
    panel_id: prefillData.panel_id,
    hgnc_id: prefillData.hgnc_id,
    source: "Literature",
    rating: prefillData.rating,
    moi: prefillData.moi,
    publications: prefillData.publications,
    phenotypes: prefillData.phenotypes,
    comments: prefillData.comments,
  };

  // Add mode_of_pathogenicity only if present
  if (prefillData.mode_of_pathogenicity) {
    fields.mode_of_pathogenicity = prefillData.mode_of_pathogenicity;
  }

  // Create hidden inputs for all fields
  Object.keys(fields).forEach(function (name) {
    var input = document.createElement("input");
    input.type = "hidden";
    input.name = name;
    input.value = fields[name];
    form.appendChild(input);
  });

  // Submit and clean up
  document.body.appendChild(form);
  form.submit();
  document.body.removeChild(form);
}

/**
 * Bind click handlers to all prefill buttons.
 */
function initPrefillButtons() {
  document.querySelectorAll(".prefill-button[data-prefill]").forEach(function (button) {
    button.addEventListener("click", handlePrefillClick);
  });
}

// Auto-initialize when DOM is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", function () {
    LiteratureAssignments.init();
    initPrefillButtons();
  });
} else {
  LiteratureAssignments.init();
  initPrefillButtons();
}
